from .build import arch_dict, dataset_dict
from .utils.distillation import train_student
from .utils.misc import Logger
# 引入基础神经元和梯度函数
from .build.architectures.snn.neurons import IFNeuron, TriangleSG 

import os
import torch
import torch.nn as nn
import argparse
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import random_seed, CheckpointSaver

# ==============================================================================
# 1. 定义 Always Zero 神经元
# ==============================================================================
class AlwaysZeroIFNeuron(IFNeuron):
    """
    修改版的 IFNeuron。
    特性：在每个时间步开始前，强制将膜电位 Vmem 清零。
    作用：消除静态输入下的膜电位累积效应。
    """
    def __init__(self, timesteps, original_vth):
        super().__init__(timesteps)
        if isinstance(original_vth, torch.nn.Parameter):
            self.vth = original_vth
        else:
            self.vth = nn.Parameter(original_vth.data.clone())
        self.sg_func = TriangleSG.apply

    def forward(self, input):
        # 归一化输入 (Relative to Threshold)
        input_t = input.reshape(self.timesteps, -1, *input.shape[1:]) / self.vth

        spks = []
        # 初始化 mem
        mem = torch.zeros_like(input_t[0])

        for t in range(self.timesteps):
            # === 核心修改: 强制 Reset ===
            mem = torch.zeros_like(input_t[0])
            
            # 积分
            mem = mem + input_t[t]
            
            # 发放脉冲
            spk = self.sg_func(mem)
            
            # Reset
            mem = mem - spk
            
            spks.append(spk)

        return torch.cat(spks, dim=0) * self.vth

# ==============================================================================
# 2. 替换辅助函数
# ==============================================================================
def replace_first_layer_neuron(model):
    print(f"[Model Surgery] Replacing first layer neuron with AlwaysZeroIFNeuron...")
    
    replaced = False
    
    # 假设模型结构是 ResNet 风格
    if hasattr(model, 'layers') and len(model.layers) > 0:
        first_block = model.layers[0]
        if hasattr(first_block, 'act1') and isinstance(first_block.act1, IFNeuron):
            original_neuron = first_block.act1
            
            new_neuron = AlwaysZeroIFNeuron(
                timesteps=original_neuron.timesteps, 
                original_vth=original_neuron.vth
            )
            
            first_block.act1 = new_neuron
            print(f"  >>> Successfully replaced: model.layers[0].act1")
            replaced = True
            
    if not replaced:
        print("[Warning] Failed to locate 'layers[0].act1'. Checking for other patterns...")
        for name, module in model.named_children():
             if "act" in name and isinstance(module, IFNeuron):
                 print(f"  >>> Fallback replaced: {name}")
                 new_neuron = AlwaysZeroIFNeuron(module.timesteps, module.vth)
                 setattr(model, name, new_neuron)
                 replaced = True
                 break 

    if not replaced:
        raise RuntimeError("Could not find any IFNeuron to replace in the first layer!")

# ==============================================================================
# 3. 主逻辑
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 注意：使用 torchrun 时，显卡选择通常通过 CUDA_VISIBLE_DEVICES 环境变量控制
    parser.add_argument('-g', '--gpus', help='select GPUs', type=str, default='0,1,2,3')
    parser.add_argument('-a', '--arch', help='architecture name', type=str, default=None)
    parser.add_argument('-d', '--dataset', help='dataset name', type=str, default=None)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=128)
    parser.add_argument('-e', '--num_epochs', help='epoch number', type=int, default=200)
    parser.add_argument('-s', '--from_scratch', help='train from scratch', action="store_true", default=False)
    parser.add_argument('-lr', '--learning_rate', help='base learning rate', type=float, default=1e-2)
    parser.add_argument('-ts', '--teacher_spiking', help='snn teacher', action="store_true", default=False)
    parser.add_argument('-tt', '--teacher_timesteps', help='teacher timesteps', type=int, default=4)
    parser.add_argument('-tp', '--teacher_precision', help='teacher bit precision', type=int, default=4)
    parser.add_argument('-ss', '--student_spiking', help='snn student', action="store_true", default=False)
    parser.add_argument('-sp', '--student_precision', help='student bit precision', type=int, default=4)
    parser.add_argument('-st', '--student_timesteps', help='student timesteps', type=int, default=4)
    parser.add_argument('-ld', '--log_dir', help='log directory', type=str, default=None)
    parser.add_argument('-dd', '--data_dir', help='dataset directory', type=str, default=None)
    parser.add_argument('-cpt', '--checkpoint_path', help='path to teacher checkpoint', type=str, default=None)
    
    parser.add_argument('--enable_always_zero', action='store_true', help='Enable always zero reset for the first layer')
    
    return parser.parse_args()


def _load_from_teacher(student_state_dict: dict, teacher_state_dict: dict) -> dict:
    for s_key in list(student_state_dict.keys()):
        if 'scale' not in s_key:
            if s_key in teacher_state_dict:
                student_state_dict[s_key] = teacher_state_dict.pop(s_key)
    return student_state_dict


def main():
    args = _parse_args()
    
    # -------- [关键修改] torchrun 适配逻辑 --------
    # torchrun 会自动设置 LOCAL_RANK, RANK, WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 初始化进程组 (不需要手动指定 rank/world_size)
    dist.init_process_group(backend='nccl')
    
    # 设定当前进程使用的 GPU
    torch.cuda.set_device(local_rank)
    random_seed(rank=rank)
    
    print(f"Process {rank} (Local Rank: {local_rank}) initialized.")
    # -----------------------------------------------

    arch_name = args.arch
    dataset_name = args.dataset
    teacher_spiking = args.teacher_spiking
    teacher_bits = args.teacher_precision
    teacher_timesteps = args.teacher_timesteps
    student_spiking = args.student_spiking
    student_bits = args.student_precision
    student_timesteps = args.student_timesteps
    batch_size = args.batch_size
    data_dir = args.data_dir
    log_dir = args.log_dir
    checkpoint_path = args.checkpoint_path

    # ----------------          create model        ----------------
    teacher = arch_dict(teacher_spiking, teacher_bits, teacher_timesteps, arch_name, dataset_name)
    student = arch_dict(student_spiking, student_bits, student_timesteps, arch_name, dataset_name)
    
    # === 手术替换 ===
    if args.enable_always_zero:
        replace_first_layer_neuron(student)
    # =================

    # 移动到对应的 GPU (使用 local_rank)
    teacher = teacher.to(device=local_rank)
    student = student.to(device=local_rank)

    # ----------------      create dataloader       ----------------
    train_set, test_set = dataset_dict(dataset_name, arch_name, data_dir)
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, num_workers=4, persistent_workers=True, pin_memory=True)

    # ----------------        create logger         ----------------
    if rank == 0:
        setup_str = (
            "\nSetup >>>\n"
            f"{'architecture':<20}{arch_name:>15}\n"
            f"{'dataset':<20}{dataset_name:>15}\n"
            f"{'teacher spiking':<20}{'True' if teacher_spiking else 'False':>15}\n"
            f"{'student spiking':<20}{'True' if student_spiking else 'False':>15}\n"
            f"{'always zero L1':<20}{'True' if args.enable_always_zero else 'False':>15}\n"
            "<<<\n\n"
        )
        exp_info = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}_always_zero"
        exp_dir = os.path.join(log_dir, exp_info)
        os.makedirs(exp_dir, exist_ok=True)
        logger = Logger(os.path.join(exp_dir, "summary.log"))
        tb_writer = SummaryWriter(os.path.join(exp_dir, "tensorboard"))
        print(setup_str)
        logger.timestamp("START TIME")
        logger.logging(setup_str)
    else:
        logger = None
        tb_writer = None
    
    # ==============================================================
    optimizer = create_optimizer_v2(
        student,
        opt="sgd",
        lr=args.learning_rate,
        weight_decay=5e-4,
        momentum=0.9
    )
    scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=args.num_epochs,
        warmup_epochs=10,
        warmup_lr=1e-5,
        cooldown_epochs=10,
        min_lr=1e-5
    )

    if rank == 0:
        checkpoint_dir = os.path.join(exp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_saver = CheckpointSaver(
            model=student,
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            decreasing=False, 
            max_history=5
        )
    else:
        checkpoint_saver = None

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(local_rank))
        teacher.load_state_dict(checkpoint['state_dict'])
    else:
        print(f"[Warning] Teacher checkpoint not found: {checkpoint_path}")

    if not args.from_scratch:
        print(f"Loading student weights from teacher...")
        student.load_state_dict(_load_from_teacher(student.state_dict(), teacher.state_dict()), strict=False)

    # DDP 包装
    student = DDP(student, device_ids=[local_rank])
        
    train_student(
        device=local_rank, # 传入 local_rank 作为 device
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        tb_writer=tb_writer,
        cpt_saver=checkpoint_saver
    )

    if rank == 0:
        print("Training done.")
        logger.logging("Training done.")
        logger.timestamp("END TIME")
        logger.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    # 彻底移除了 mp.spawn
    main()