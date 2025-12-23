import argparse
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import random_seed

# Import your existing modules
from .build import arch_dict, dataset_dict
from .utils.distillation import DistillKL, CosineAnnealRestart
from .utils.misc import Logger, AverageMeter, BAR_FMT
from .utils.evaluation import evaluate_model
from .build.architectures.quant_layers import QuantConv2d, QuantLinear

def _parse_args():
    parser = argparse.ArgumentParser()
    # Basic Config
    parser.add_argument('-g', '--gpus', type=str, default='0')
    parser.add_argument('-a', '--arch', type=str, default='resnet20')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('-e', '--num_epochs', type=int, default=200)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-2)
    parser.add_argument('-ld', '--log_dir', type=str, default='./logs_prune')
    parser.add_argument('-dd', '--data_dir', type=str, default='./data')
    parser.add_argument('--from_scratch', action='store_true', help='Do not init from teacher')
    
    # DDP Config (新增)
    parser.add_argument('--master_port', type=str, default='12345', help='DDP Master Port')
    
    # Teacher (Optional but recommended for recovery)
    parser.add_argument('-cpt', '--teacher_checkpoint', type=str, default=None)
    parser.add_argument('--teacher_spiking', action="store_true")
    parser.add_argument('--teacher_timesteps', type=int, default=4)
    parser.add_argument('--teacher_precision', type=int, default=32)

    # Student (Target Model)
    parser.add_argument('--student_spiking', action="store_true", default=True)
    parser.add_argument('--student_timesteps', type=int, default=4)
    parser.add_argument('--student_precision', type=int, default=32, help='32 for FP32 weights')
    parser.add_argument('--student_checkpoint', type=str, default=None, help='Load existing student weights')

    # Pruning Config
    parser.add_argument('--target_sparsity', type=float, default=0.85, help='Target sparsity (e.g. 0.85)')
    parser.add_argument('--prune_start_epoch', type=int, default=5)
    parser.add_argument('--prune_end_epoch', type=int, default=150)
    parser.add_argument('--prune_freq', type=int, default=1)

    return parser.parse_args()

def _load_from_teacher(student_state_dict: dict, teacher_state_dict: dict) -> dict:
    for s_key in list(student_state_dict.keys()):
        if s_key in teacher_state_dict and 'scale' not in s_key:
            student_state_dict[s_key] = teacher_state_dict[s_key]
    return student_state_dict

class Pruner:
    def __init__(self, model, target_sparsity, start_epoch, end_epoch, freq):
        self.model = model
        self.target_sparsity = target_sparsity
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.freq = freq
        self.modules = []
        
        # Identify layers to prune (QuantConv2d and QuantLinear)
        for name, m in model.named_modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                self.modules.append((m, 'weight'))
                
    def step(self, epoch):
        if epoch < self.start_epoch or epoch > self.end_epoch:
            return
        if (epoch - self.start_epoch) % self.freq != 0:
            return
            
        # Cubic Schedule
        total_steps = self.end_epoch - self.start_epoch
        current_step = epoch - self.start_epoch
        progress = 1.0 - (current_step / total_steps)
        current_sparsity = self.target_sparsity * (1 - progress ** 3)
        
        for m, name in self.modules:
            prune.l1_unstructured(m, name=name, amount=current_sparsity)
            
        print(f"[Pruner] Epoch {epoch}: Sparsity updated to {current_sparsity:.4f}")

    def make_permanent(self):
        print("[Pruner] Removing pruning masks (making sparsity permanent)...")
        for m, name in self.modules:
            if prune.is_pruned(m):
                prune.remove(m, name)

def main(rank, args):
    # DDP Setup
    gpu_ids = list(map(int, args.gpus.split(',')))
    os.environ['MASTER_ADDR'] = 'localhost'
    # 使用传入的端口，不再硬编码
    os.environ['MASTER_PORT'] = args.master_port
    
    dist.init_process_group(backend='nccl', rank=rank, world_size=len(gpu_ids))
    torch.cuda.set_device(rank)
    random_seed(rank)

    # Models
    student = arch_dict(args.student_spiking, args.student_precision, args.student_timesteps, args.arch, args.dataset)
    student = student.cuda(rank)
    
    teacher = None
    if args.teacher_checkpoint:
        teacher = arch_dict(args.teacher_spiking, args.teacher_precision, args.teacher_timesteps, args.arch, args.dataset)
        teacher = teacher.cuda(rank)
        ckpt = torch.load(args.teacher_checkpoint, map_location={'cuda:0': f'cuda:{rank}'})
        
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        teacher.load_state_dict(new_state_dict, strict=False)
        teacher.eval()
        
        if not args.from_scratch and args.student_checkpoint is None:
            student.load_state_dict(_load_from_teacher(student.state_dict(), teacher.state_dict()), strict=False)
            if rank == 0: print("Student initialized from Teacher weights.")

    if args.student_checkpoint:
        ckpt = torch.load(args.student_checkpoint, map_location={'cuda:0': f'cuda:{rank}'})
        student.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt, strict=False)
        if rank == 0: print(f"Student loaded from {args.student_checkpoint}")

    student = DDP(student, device_ids=[rank])
    pruner = Pruner(student.module, args.target_sparsity, args.prune_start_epoch, args.prune_end_epoch, args.prune_freq)

    # Data
    train_set, test_set = dataset_dict(args.dataset, args.arch, args.data_dir)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=DistributedSampler(train_set, rank=rank), num_workers=4)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, sampler=DistributedSampler(test_set, rank=rank), num_workers=4)

    # Optimizer
    optimizer = create_optimizer_v2(student, opt="sgd", lr=args.learning_rate, weight_decay=1e-4, momentum=0.9)
    scheduler, num_epochs = create_scheduler_v2(optimizer, sched="cosine", num_epochs=args.num_epochs, warmup_epochs=5)
    
    dist_criterion = DistillKL(temperature=4.0)
    
    logger = None
    tb_writer = None
    if rank == 0:
        exp_name = f"prune_fp32_{args.target_sparsity}_{datetime.now().strftime('%m%d_%H%M')}"
        log_dir = os.path.join(args.log_dir, exp_name)
        os.makedirs(log_dir, exist_ok=True)
        logger = Logger(os.path.join(log_dir, "train.log"))
        tb_writer = SummaryWriter(os.path.join(log_dir, "tb"))
        print(f"Start pruning training. Logs: {log_dir}")

    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        student.train()
        pruner.step(epoch)

        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        
        pbar = tqdm(train_loader, disable=(rank!=0), desc=f"Ep {epoch}")
        for x, y in pbar:
            x, y = x.cuda(rank), y.cuda(rank)
            out_s = student(x)
            loss = F.cross_entropy(out_s, y)
            
            if teacher:
                with torch.no_grad(): out_t = teacher(x)
                loss += 2.0 * dist_criterion(out_s, out_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            acc = (out_s.argmax(1) == y).float().mean()
            dist.all_reduce(loss)
            dist.all_reduce(acc)
            loss_meter.update(loss.item()/dist.get_world_size(), x.size(0))
            acc_meter.update(acc.item()/dist.get_world_size(), x.size(0))
            pbar.set_postfix(loss=loss_meter.avg, acc=acc_meter.avg)
            
        scheduler.step(epoch)
        val_acc = evaluate_model(rank, student, test_loader)
        
        if rank == 0:
            logger.logging(f"Epoch {epoch} | Loss {loss_meter.avg:.4f} | Val Acc {val_acc:.4f}")
            if tb_writer:
                tb_writer.add_scalar('Acc/Val', val_acc, epoch)
                tb_writer.add_scalar('Sparsity', pruner.target_sparsity if epoch >= pruner.end_epoch else 0, epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(student.module.state_dict(), os.path.join(log_dir, "best_masked.pth"))

    if rank == 0:
        print("Finalizing pruning...")
        if os.path.exists(os.path.join(log_dir, "best_masked.pth")):
            student.module.load_state_dict(torch.load(os.path.join(log_dir, "best_masked.pth")))
        
        pruner.make_permanent()
        
        zeros = 0
        total = 0
        for m in student.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                zeros += (m.weight == 0).sum().item()
                total += m.weight.numel()
        print(f"Final Global Sparsity: {zeros/total:.2%}")
        
        save_path = os.path.join(log_dir, "final_pruned_fp32.pth")
        torch.save({'state_dict': student.module.state_dict(), 'acc': best_acc}, save_path)
        print(f"Model saved to {save_path}. Ready for export.")

    dist.destroy_process_group()

if __name__ == '__main__':
    args = _parse_args()
    torch.multiprocessing.spawn(main, args=(args,), nprocs=len(args.gpus.split(',')))