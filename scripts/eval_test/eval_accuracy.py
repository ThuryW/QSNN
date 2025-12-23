import argparse
import os
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # [新增] 引入 DDP
from torch.utils.data import DataLoader, DistributedSampler
from timm.utils import random_seed

# ==============================================================================
# 环境设置：将工程根目录加入路径，以便导入 engine 模块
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from engine.build import arch_dict, dataset_dict
from engine.utils.evaluation import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate FP32 Model Accuracy')
    
    # 基础配置
    parser.add_argument('-g', '--gpus', default='0', type=str, help='Visible GPUs, e.g. "0,1"')
    parser.add_argument('-dd', '--data_dir', default='/home/wangtianyu/dataset', type=str, help='Path to dataset')
    parser.add_argument('-b', '--batch_size', default=128, type=int, help='Batch size per GPU')
    
    # 模型配置 (无需 bits 参数，默认为 32)
    parser.add_argument('-a', '--arch', default='resnet20', type=str, help='Model architecture')
    parser.add_argument('-d', '--dataset', default='cifar10', type=str, help='Dataset name')
    parser.add_argument('-s', '--spiking', action='store_true', help='Enable SNN mode')
    parser.add_argument('-t', '--timesteps', default=4, type=int, help='SNN timesteps (ignored for ANN)')
    
    # 权重文件
    parser.add_argument('-cpt', '--checkpoint', type=str, required=True, help='Path to checkpoint (.pth/.tar)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --------------------------------------------------------------------------
    # 1. 初始化分布式环境
    # --------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    device = local_rank
    
    # [修改] 与 main_dist.py 保持完全一致的随机种子设置逻辑
    random_seed(rank=local_rank)

    if local_rank == 0:
        print(f"\n>>> [Evaluation] Full Precision (FP32) Mode")
        print(f"Arch: {args.arch} | Dataset: {args.dataset}")
        print(f"Type: {'SNN (T={args.timesteps})' if args.spiking else 'ANN'}")
        print(f"Checkpoint: {args.checkpoint}")

    # --------------------------------------------------------------------------
    # 2. 构建模型 (强制 bits=32)
    # --------------------------------------------------------------------------
    try:
        model = arch_dict(
            spiking=args.spiking,
            bits=32,  # 强制全精度
            timesteps=args.timesteps,
            arch_name=args.arch,
            dataset_name=args.dataset
        )
    except Exception as e:
        print(f"[Error] Model build failed: {e}")
        return

    model = model.to(device)

    # --------------------------------------------------------------------------
    # 3. 加载权重
    # --------------------------------------------------------------------------
    if os.path.isfile(args.checkpoint):
        if local_rank == 0:
            print(f">>> Loading weights...")
        
        checkpoint = torch.load(args.checkpoint, map_location=torch.device(f'cuda:{local_rank}'))
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # 移除 DDP 产生的 'module.' 前缀 (先加载到纯模型，再包 DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=True)
    else:
        print(f"[Error] Checkpoint file not found: {args.checkpoint}")
        return

    # [新增] 将模型包装为 DDP
    # 这是为了与 main_dist.py 中的 student = DDP(...) 保持完全一致的结构
    model = DDP(model, device_ids=[local_rank])

    # --------------------------------------------------------------------------
    # 4. 数据加载
    # --------------------------------------------------------------------------
    # dataset_dict 内部调用 get_dataset，包含 transforms 定义
    _, test_set = dataset_dict(args.dataset, args.arch, args.data_dir)
    
    # DistributedSampler 负责在多卡间切分数据（并进行 Padding 以保证整除）
    sampler = DistributedSampler(test_set, num_replicas=world_size, rank=local_rank, shuffle=False)
    
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=4, 
        persistent_workers=True, # [新增] 与 main_dist.py 保持一致
        pin_memory=True
    )

    # --------------------------------------------------------------------------
    # 5. 推理评估
    # --------------------------------------------------------------------------
    # 调用与蒸馏过程中完全相同的评估函数
    acc = evaluate_model(device, model, test_loader)
    
    if local_rank == 0:
        print(f"\n{'='*40}")
        print(f"Final Evaluation Accuracy: {acc * 100:.2f}%")
        print(f"{'='*40}\n")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()