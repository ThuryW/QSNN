import torch
import torch.nn as nn
from tqdm import tqdm
from .misc import AverageMeter, BAR_FMT
from typing import Optional
import torch.distributed as dist


def evaluate_model(
    device,
    model: nn.Module,
    test_loader: str = None,
    num_batches: Optional[int] = None
) -> float:
    
    model.eval()
    acc = AverageMeter()
    num_batches = len(test_loader) if num_batches is None else min(len(test_loader), num_batches)

    pbar = tqdm(
        iterable=test_loader,
        total=num_batches,
        desc='evaluating',
        bar_format=BAR_FMT,
        disable=(device != 0),
        leave=False
    )
    with torch.no_grad():
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            batch_acc = (outputs.argmax(dim=1) == targets).float().sum().item() / targets.shape[0]
            acc.update(batch_acc)
            pbar.set_postfix({'batch acc': f"{batch_acc*100:06.2f}%"})
    
    acc = torch.tensor([acc.avg], device=device)
    dist.all_reduce(acc, op=dist.ReduceOp.AVG)
    return acc.item()