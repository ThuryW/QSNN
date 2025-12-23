import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# class WeightFakeQuantizer(nn.Module):
#     """
#     `x_q` = round ( clamp ( (`x` / `s`, `qmin`, `qmax`) )\n
#     `x_fq` = `x_q` * `s`
#     """
#     def __init__(
#         self,
#         bits: int,
#         num_channels: int = None
#     ):
#         super().__init__()
#         self.bits = bits
        
#         self.qmin = -2 ** (bits - 1)
#         self.qmax = 2 ** (bits - 1) - 1
        
#         self.min_val: nn.Buffer
#         self.register_buffer('min_val', torch.zeros(num_channels))
#         self.max_val: nn.Buffer
#         self.register_buffer('max_val', torch.zeros(num_channels))

#         self.initialized = False

#     def extra_repr(self):
#         return f"bits={self.bits}"
    
#     @torch.no_grad()
#     def mav_update(self, x: torch.Tensor):
#         x = x.flatten(1)
#         x_min = x.min(dim=1).values
#         x_max = x.max(dim=1).values

#         if dist.is_initialized():
#             dist.all_reduce(x_min, op=dist.ReduceOp.AVG)
#             dist.all_reduce(x_max, op=dist.ReduceOp.AVG)

#         if not self.initialized:
#             self.min_val.copy_(x_min)
#             self.max_val.copy_(x_max)
#             self.initialized = True
#         else:
#             self.min_val.copy_(0.9 * self.min_val + 0.1 * x_min)
#             self.max_val.copy_(0.9 * self.max_val + 0.1 * x_max)

#     def get_params(self) -> torch.Tensor:
#         scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
#         return scale

#     def forward(self, x: torch.Tensor):
#         if self.training:
#             self.mav_update(x)
        
#         s = self.get_params()
#         s = s.reshape(x.shape[0], *([1] * (x.dim() - 1)))

#         _x_q = torch.clamp(x / s, self.qmin, self.qmax)
#         x_q = _x_q + (_x_q.round() - _x_q).detach()
        
#         x_fq = x_q * s

#         return x_fq  


class WeightFakeQuantizer(nn.Module):
    """
    `x_q` = round ( clamp ( (`x` / `s`, `qmin`, `qmax`) )\n
    `x_fq` = `x_q` * `s`
    """
    def __init__(
        self,
        bits: int,
        num_channels: int = None
    ):
        super().__init__()
        self.bits = bits
        
        self.qmin = -2 ** (bits - 1)
        self.qmax = 2 ** (bits - 1) - 1
        
        self.scale = nn.Parameter(torch.ones(num_channels))
        self.initialized = False

    def extra_repr(self):
        return f"q_min={self.qmin}, q_max={self.qmax}"
    
    @torch.no_grad()
    def init_scale(self, x: torch.Tensor):
        x = x.flatten(1)
        x_mean, x_std = x.mean(dim=1), x.std(dim=1)

        if dist.is_initialized():
            dist.all_reduce(x_mean, op=dist.ReduceOp.AVG)
            dist.all_reduce(x_std, op=dist.ReduceOp.AVG)

        scale_init = torch.where(x_mean > 0, x_mean + 3 * x_std, x_mean - 3 * x_std) / (self.qmax + 1)
        self.scale.copy_(scale_init)
        self.initialized = True

    def forward(self, x: torch.Tensor):
        if self.training and not self.initialized:
            self.init_scale(x)
        
        grad_scaler = 1 / (x.numel() / x.shape[0] * self.qmax) ** 0.5
        s = self.scale * grad_scaler + (self.scale - self.scale * grad_scaler).detach()
        s = self.scale.reshape(x.shape[0], *([1] * (x.dim() - 1)))

        _x_q = torch.clamp(x / s, self.qmin, self.qmax)
        x_q = _x_q + (_x_q.round() - _x_q).detach()
        
        x_fq = x_q * s

        return x_fq 


class QuantConv2d(nn.Conv2d):

    def __init__(
        self,
        bits: int,
        in_channels,
        out_channels,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1,
        groups = 1,
        bias = True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
        self.bits = bits
        self.w_quantizer = WeightFakeQuantizer(bits, out_channels) if bits < 32 else None

    def extra_repr(self):
        return f"bits={self.bits}"

    def forward(self, input: torch.Tensor):
        w_fq = self.w_quantizer(self.weight) if self.bits < 32 else self.weight
        output = F.conv2d(input, w_fq, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output
    
    
class QuantLinear(nn.Linear):

    def __init__(
        self,
        bits: int,
        in_features,
        out_features,
        bias = True
    ):
        super().__init__(in_features, out_features, bias)
        self.bits = bits
        self.w_quantizer = WeightFakeQuantizer(bits, out_features) if bits < 32 else None

    def extra_repr(self):
        return f"bits={self.bits}"

    def forward(self, input: torch.Tensor):
        w_fq = self.w_quantizer(self.weight) if self.bits < 32 else self.weight
        output = F.linear(input, w_fq, self.bias)
        return output