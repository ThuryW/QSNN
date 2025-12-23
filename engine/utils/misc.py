import time
import torch
import torch.nn as nn
from typing import Optional


BAR_FMT = (
    '{desc:<18}{percentage:3.0f}%|'
    '{bar:9}|{n_fmt:>6}/{total_fmt:<6}|'
    '{elapsed:>8}<{remaining:<8}{postfix:>20}|'
)


class AverageMeter():
    """
    Computes average value.
    Attributes:
        val: current value
        avg: average value
        sum: summation of all the updated values
        count: update count
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, step=1):
        self.val = val
        self.sum += val * step
        self.count += step
        self.avg = self.sum / self.count


class Logger():

    def __init__(
        self,
        log_file: Optional[str] = None
    ):
        if log_file is not None:
            self.file = open(log_file, 'w')
        else:
            self.file = None
    
    def logging(self, message):
        if self.file:
            self.file.write(message + '\n')
            self.file.flush()

    def timestamp(self, message):
        timestamp = f"{message}: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
        return self.logging(timestamp)

    def close(self):
        if self.file:
            self.file.close()