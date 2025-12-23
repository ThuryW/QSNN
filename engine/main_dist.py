from .build import arch_dict, dataset_dict
from .utils.distillation import train_student
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
    return parser.parse_args()


def _load_from_teacher(student_state_dict: dict, teacher_state_dict: dict) -> dict:
    for s_key in list(student_state_dict.keys()):
        if 'scale' not in s_key:
            student_state_dict[s_key] = teacher_state_dict.pop(s_key)
    return student_state_dict


def main(rank, args):
    # ----------------       parse arguments        ----------------
    gpu_ids = list(map(int, args.gpus.split(',')))
    world_size = len(args.gpus.split(','))
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

    # --------      init distributed training process       --------    
    torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    random_seed(rank=rank)
    print(f"process {rank} using GPU {gpu_ids[rank]}")

    # ----------------          create model        ----------------
    teacher = arch_dict(teacher_spiking, teacher_bits, teacher_timesteps, arch_name, dataset_name)
    student = arch_dict(student_spiking, student_bits, student_timesteps, arch_name, dataset_name)
    teacher = teacher.to(device=rank)
    student = student.to(device=rank)

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
            f"{'teacher weight bits':<20}{teacher_bits:>15}\n"
            f"{'teacher timesteps':<20}{teacher_timesteps:>15}\n"
            f"{'student spiking':<20}{'True' if student_spiking else 'False':>15}\n"
            f"{'student weight bits':<20}{student_bits:>15}\n"
            f"{'student timesteps':<20}{student_timesteps:>15}\n"
            "<<<\n\n"
            f"Teacher arch >>>\n{teacher}\n<<<\n\n"
            f"Student arch >>>\n{student}\n<<<\n"
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

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(rank))
    teacher.load_state_dict(checkpoint['state_dict'])

    if not args.from_scratch:
        student.load_state_dict(_load_from_teacher(student.state_dict(), teacher.state_dict()))

    student = DDP(student, device_ids=[rank])
        
    train_student(
        device=rank,
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

    # ----------------        end of function       ----------------
    if rank == 0:
        logger.timestamp("END TIME")
        logger.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    args = _parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    torch.multiprocessing.spawn(fn=main, args=(args,), nprocs=len(args.gpus.split(',')))
