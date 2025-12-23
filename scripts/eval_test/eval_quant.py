import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from engine.build import dataset_dict
from engine.build.architectures.snn.resnet import spk_resnet20
from engine.build.architectures.snn.vgg import spk_vgg11

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def smart_load_model(model, checkpoint_path):
    print(f"==> Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location='cpu') # 先加载到 CPU
    
    # 获取 state_dict
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # 1. 检查 Checkpoint 中的 key 格式
    ckpt_keys = list(state_dict.keys())
    print(f"    [Debug] First checkpoint key: {ckpt_keys[0]}")
    
    # 2. 检查 Model 中的 key 格式
    model_keys = list(model.state_dict().keys())
    print(f"    [Debug] First model key:      {model_keys[0]}")

    # 3. 处理 'module.' 前缀不一致问题
    # 策略：我们将 checkpoint 的 key 统一处理成与 model 匹配的格式
    new_state_dict = {}
    
    model_has_module = model_keys[0].startswith("module.")
    ckpt_has_module = ckpt_keys[0].startswith("module.")

    for k, v in state_dict.items():
        new_key = k
        if ckpt_has_module and not model_has_module:
            new_key = k.replace("module.", "")
        elif not ckpt_has_module and model_has_module:
            new_key = "module." + k
        new_state_dict[new_key] = v

    # 4. 尝试加载并捕捉不匹配的 key
    # strict=False 但我们会手动打印 missing_keys
    msg = model.load_state_dict(new_state_dict, strict=False)
    
    print(f"==> Load Result: {msg}")
    
    if len(msg.missing_keys) > 0:
        print("\n!!!!!!!! WARNING: MISSING KEYS !!!!!!!!")
        # 过滤掉 num_batches_tracked 这种无关紧要的 key
        real_missing = [k for k in msg.missing_keys if "num_batches_tracked" not in k]
        if len(real_missing) > 0:
            print(f"Critical missing keys (First 10): {real_missing[:10]}")
            if "weight" in str(real_missing):
                raise RuntimeError("Critical weights failed to load! Check architecture mismatch.")
        else:
            print("Only 'num_batches_tracked' missing (Safe to ignore for BN).")
            
    if len(msg.unexpected_keys) > 0:
        print(f"\n!!!!!!!! WARNING: UNEXPECTED KEYS (First 10): {msg.unexpected_keys[:10]}")

    return checkpoint.get('epoch', 'Unknown')

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            prec1 = accuracy(output.data, target, topk=(1,))[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Final Accuracy: {top1.avg:.3f}%')
    return top1.avg

def main():
    parser = argparse.ArgumentParser(description='SNN Quantization Testing')
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--arch', default='resnet20', type=str)
    parser.add_argument('--resume', required=True, type=str, metavar='PATH') # 强制要求 resume
    parser.add_argument('--precision', default=32, type=int)
    parser.add_argument('--timesteps', default=4, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=4, type=int)
    
    args = parser.parse_args()

    # 1. 构建数据集
    print(f"==> Preparing data: {args.dataset}")
    train_set, test_set = dataset_dict(args.dataset, args.arch, args.data_dir)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 2. 构建模型
    print(f"==> Creating model: {args.arch} | Bits: {args.precision} | Timesteps: {args.timesteps}")
    if 'resnet20' in args.arch:
        num_classes = 10 if args.dataset == 'cifar10' else 100
        # 确保这里的调用参数与您 quant_layers.py 修改后的接口一致
        model = spk_resnet20(num_classes=num_classes, timesteps=args.timesteps, bits=args.precision)
    elif 'vgg11' in args.arch:
        num_classes = 10 if args.dataset == 'cifar10' else 100
        # 确保这里的调用参数与您 quant_layers.py 修改后的接口一致
        model = spk_vgg11(num_classes=num_classes, timesteps=args.timesteps, bits=args.precision)
    else:
        raise ValueError("Only resnet20 and vgg11 supported in this script.")

    # 3. 验证加载前后的权重变化 (Sanity Check)
    # 获取第一层卷积的权重的 sum 作为指纹
    param_fingerprint_before = list(model.parameters())[0].sum().item()

    model = model.cuda()
    # 注意：这里我们先不加 DataParallel，加载完权重后再加，减少 module. 前缀的困扰
    # 或者如果权重本身就是 DataParallel 产生的，我们在 smart_load_model 里处理

    # 4. 加载权重
    epoch = smart_load_model(model, args.resume)
    
    param_fingerprint_after = list(model.parameters())[0].sum().item()
    if param_fingerprint_before == param_fingerprint_after:
        print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("CRITICAL ERROR: Weights did not change after loading!")
        print("The model is still using random initialization.")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    else:
        print(f"==> Weights updated successfully (Fingerprint: {param_fingerprint_before:.4f} -> {param_fingerprint_after:.4f})")

    # 5. 为了推理加速，现在可以包裹 DataParallel
    model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss().cuda()

    # 6. 执行测试
    print("==> Starting evaluation...")
    validate(val_loader, model, criterion)

if __name__ == '__main__':
    main()