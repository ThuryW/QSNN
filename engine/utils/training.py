from .misc import AverageMeter, BAR_FMT
from .evaluation import evaluate_model
import torch
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F


def train_model(
    device,
    model,
    train_loader,
    test_loader,
    num_epochs,
    optimizer,
    scheduler,
    loss_scaler,
    logger = None,
    tb_writer = None,
    cpt_saver = None
) -> None:
    
    if device == 0:
        logger.logging(
            "Training records >>>\n"
            f"{'epoch':>5}|{'lr':>10}|"
            f"{'task loss':>10}|{'eval acc':>10}|{'best acc':>10}|"
        )
    
    task_loss = AverageMeter()

    pbar_proc = tqdm(
        total=num_epochs,
        desc=f"training",
        bar_format=BAR_FMT,
        disable=(device != 0),
        leave=False
    )
    for epoch in range(num_epochs):
        task_loss.reset()
        train_loader.sampler.set_epoch(epoch)
        model.train()

        pbar_epoch = tqdm(
            iterable=train_loader,
            total=len(train_loader),
            desc=f"epoch {epoch + 1}",
            bar_format=BAR_FMT, 
            disable=(device != 0),
            leave=False
        )
        for inputs, targets in pbar_epoch:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                _task_loss = F.cross_entropy(outputs, targets)
                
            optimizer.zero_grad()
            loss_scaler.scale(_task_loss).backward()
            loss_scaler.step(optimizer)
            loss_scaler.update()
      
            dist.all_reduce(_task_loss, op=dist.ReduceOp.AVG)
            task_loss.update(_task_loss.item())

            pbar_epoch.set_postfix({
                'loss': f"{_task_loss:05.3f}"
            })

        scheduler.step(epoch + 1)

        val_acc = evaluate_model(device, model, test_loader)
        
        if device == 0:
            tb_writer.add_scalar(f"train_task_loss", task_loss.avg, epoch)
            tb_writer.add_scalar(f"eval_acc", val_acc, epoch)
            best_acc, _ = cpt_saver.save_checkpoint(epoch + 1, val_acc)
            logger.logging(
                f"{epoch + 1:>5}|{optimizer.param_groups[0]['lr']:>10.5f}|"
                f"{task_loss.avg:>10.5f}|{val_acc * 100:>9.2f}%|{best_acc * 100:>9.2f}%|"
            )
        pbar_proc.update(1)

    pbar_proc.clear()
        
    if device == 0:
        logger.logging("<<<\n")