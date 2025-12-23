from .misc import AverageMeter, BAR_FMT
from .evaluation import evaluate_model
import math
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.distributed as dist
import torch.nn.functional as F


class CosineAnnealRestart():

    def __init__(self, stages):
        self.stages = stages
        self.cur_stage = 0
        self.stage_step = 0
        self.value = stages[0][0]

    def step(self) -> float:
        start, end, length = self.stages[self.cur_stage]
        progress = self.stage_step / length
        progress = min(progress, 1.0)
        self.value = end + (start - end) * 0.5 * (1 + math.cos(math.pi * progress))

        self.stage_step += 1
        if self.stage_step >= length and self.cur_stage < len(self.stages) - 1:
            self.cur_stage += 1
            self.stage_step = 0

        return self.value


class DistillKL(nn.Module):
    
    def __init__(
        self,
        temperature: float = 4.
    ):
        super().__init__()
        self.temperature = temperature

    def forward(self, student_outputs, teacher_outputs) -> torch.Tensor:
        student_prob = F.log_softmax(student_outputs / self.temperature, dim=1)
        teacher_prob = F.softmax(teacher_outputs / self.temperature, dim=1)
        loss = F.kl_div(student_prob, teacher_prob, reduction='batchmean') * (self.temperature ** 2)
        return loss


def train_student(
    device,
    student,
    teacher,
    train_loader,
    test_loader,
    num_epochs,
    optimizer,
    scheduler,
    logger = None,
    tb_writer = None,
    cpt_saver = None
) -> None:
    
    if device == 0:
        logger.logging(
            "Training records >>>\n"
            f"{'epoch':>5}|{'lr':>10}|"
            f"{'task loss':>10}|{'dist loss':>10}|{'total loss':>10}|"
            f"{'eval acc':>10}|{'best acc':>10}|"
        )
    
    task_loss = AverageMeter()
    dist_loss = AverageMeter()
    total_loss = AverageMeter()

    dist_criterion = DistillKL(temperature=4.)
    lambda_dist = CosineAnnealRestart([
        (1., 1., 310)
    ])

    pbar_proc = tqdm(
        total=num_epochs,
        desc=f"training",
        bar_format=BAR_FMT,
        disable=(device != 0),
        leave=False
    )
    for epoch in range(num_epochs):
        task_loss.reset()
        dist_loss.reset()
        total_loss.reset()
        train_loader.sampler.set_epoch(epoch)
        student.train()
        teacher.eval()

        _lambda_dist = lambda_dist.step()

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

            with torch.no_grad():
                teacher_outputs = teacher(inputs)

            student_outputs = student(inputs)
            _task_loss = F.cross_entropy(student_outputs, targets)
            _dist_loss = dist_criterion(student_outputs, teacher_outputs)
            _total_loss = _task_loss + _lambda_dist * _dist_loss
                
            optimizer.zero_grad()
            _total_loss.backward()
            optimizer.step()
      
            dist.all_reduce(_task_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(_dist_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(_total_loss, op=dist.ReduceOp.AVG)
            task_loss.update(_task_loss.item())
            dist_loss.update(_dist_loss.item())
            total_loss.update(_total_loss.item())

            pbar_epoch.set_postfix({
                'loss': f"{_total_loss:05.3f}"
            })

        scheduler.step(epoch + 1)

        val_acc = evaluate_model(device, student, test_loader)
        
        if device == 0:
            tb_writer.add_scalar(f"train_task_loss", task_loss.avg, epoch)
            tb_writer.add_scalar(f"train_dist_loss", dist_loss.avg, epoch)
            tb_writer.add_scalar(f"train_total_loss", total_loss.avg, epoch)
            tb_writer.add_scalar(f"eval_acc", val_acc, epoch)
            best_acc, _ = cpt_saver.save_checkpoint(epoch + 1, val_acc)
            logger.logging(
                f"{epoch + 1:>5}|{optimizer.param_groups[0]['lr']:>10.5f}|"
                f"{task_loss.avg:>10.5f}|{dist_loss.avg:>10.5f}|{total_loss.avg:10.5f}|"
                f"{val_acc * 100:>9.2f}%|{best_acc * 100:>9.2f}%|"
            )
        pbar_proc.update(1)

    pbar_proc.clear()

    if device == 0:
        logger.logging("<<<\n\n")