from .build import arch_dict, dataset_dict
from .utils.training import train_model
from .utils.misc import Logger
import os
import torch
import argparse
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import random_seed, CheckpointSaver


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', help='select GPUs', type=str, default='0,1,2,3')
    parser.add_argument('-a', '--arch', help='architecture name', type=str, default=None)
    parser.add_argument('-d', '--dataset', help='dataset name', type=str, default=None)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=256)
    parser.add_argument('-e', '--num_epochs', help='epoch number', type=int, default=200)
    parser.add_argument('-lr', '--learning_rate', help='base learning rate', type=float, default=1e-2)
    parser.add_argument('-s', '--spiking', help='snn', action="store_true", default=False)
    parser.add_argument('-p', '--precision', help='bit precision', type=int, default=4)
    parser.add_argument('-t', '--timesteps', help='timesteps', type=int, default=4)
    parser.add_argument('-ld', '--log_dir', help='log directory', type=str, default=None)
    parser.add_argument('-dd', '--data_dir', help='dataset directory', type=str, default=None)
    parser.add_argument('--resume', help='path to checkpoint for finetuning', type=str, default=None)
    return parser.parse_args()


def main(rank, args):
    # ----------------       parse arguments        ----------------
    gpu_ids = list(map(int, args.gpus.split(',')))
    world_size = len(args.gpus.split(','))
    arch_name = args.arch
    dataset_name = args.dataset
    spiking = args.spiking
    bits = args.precision
    timesteps = args.timesteps
    batch_size = args.batch_size
    data_dir = args.data_dir
    log_dir = args.log_dir

    # --------      init distributed training process       --------    
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    random_seed(rank=rank)
    print(f"Process {rank} using GPU {gpu_ids[rank]}")

    # ----------------          create model        ----------------
    model = arch_dict(spiking, bits, timesteps, arch_name, dataset_name)
    model = model.to(device=rank)
    
    # ----------------       load checkpoint        ----------------
    if args.resume:
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(args.resume, map_location=map_location)
        # 处理可能存在的 'module.' 前缀 (如果是 DDP 保存的)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        # 加载权重，strict=False 允许在微调时有轻微结构差异（虽然这里架构应该是一样的）
        msg = model.load_state_dict(new_state_dict, strict=False)
        if rank == 0:
            print(f"Loaded checkpoint from {args.resume}. Msg: {msg}")

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
            f"{'spiking':<20}{'True' if spiking else 'False':>15}\n"
            f"{'weight bits':<20}{bits:>15}\n"
            f"{'timesteps':<20}{timesteps:>15}\n"
            "<<<\n\n"
            f"Model arch >>>\n{model}\n<<<\n"
        )
        exp_info = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}"
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
        model,
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
    loss_scaler = torch.amp.GradScaler()
    setattr(loss_scaler, 'state_dict_key', "amp_scaler")

    if rank == 0:
        checkpoint_dir = os.path.join(exp_dir, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_saver = CheckpointSaver(
            model=model,
            optimizer=optimizer,
            amp_scaler=loss_scaler,
            checkpoint_dir=checkpoint_dir,
            decreasing=False, 
            max_history=5
        )
    else:
        checkpoint_saver = None

    model = DDP(model, device_ids=[rank])
    
    train_model(
        device=rank,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_scaler=loss_scaler,
        logger=logger,
        tb_writer=tb_writer,
        cpt_saver=checkpoint_saver
    )

    if rank == 0:
        print("Training done.")
        logger.logging("Training done.")

    # ----------------        end of function       ----------------
    if rank == 0:
        logger.timestamp("END TIME")
        logger.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.multiprocessing.spawn(fn=main, args=(args,), nprocs=len(args.gpus.split(',')))
