import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightFakeQuantizer(nn.Module):
    """
    Implementation of ReScaW (Weight Rescaling Strategy) from QP-SNN paper.
    Reference: Eq (10) in QP-SNN: Quantized and Pruned Spiking Neural Networks
    """
    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits
        # 根据 QP-SNN 论文公式 (4) 和 (10)，s(b) = 2^b - 1
        # 对于 2-bit，s(b) = 3。
        self.s_b = 2 ** self.bits - 1
        # 论文中提到 z set to 1 (Eq 4 description)
        self.z = 1.0

    def extra_repr(self):
        return f"bits={self.bits}, strategy=ReScaW(L1-mean)"

    def forward(self, x: torch.Tensor):
        # 1. 计算 Gamma (Scaling Factor)
        # 论文 Section 5.4 Analysis: 1-norm mean value performs best [cite: 374]
        # gamma = ||W||_1 / |W|
        gamma = x.abs().mean()
        
        # 避免除以0
        gamma = gamma + 1e-8

        # 2. Rescale Weights (Eq 9: W_scaled = W / gamma)
        x_scaled = x / gamma

        # 3. Uniform Quantization (Eq 10)
        # core formula: 2/s(b) * round( s(b)/2 * (clamp(w/gamma, -1, 1) + z) ) - z
        
        # Clamp to [-1, 1]
        x_clamped = torch.clamp(x_scaled, -1.0, 1.0)
        
        # Quantize to integer grid [0, s(b)]
        # x_int = round( s(b)/2 * (x + z) )
        scale_factor = self.s_b / 2.0
        x_projected = scale_factor * (x_clamped + self.z)
        
        # Straight-Through Estimator (STE) for rounding
        # forward: round, backward: identity
        x_rounded = (x_projected.round() - x_projected).detach() + x_projected
        
        # De-quantize back to [-1, 1] scale
        x_quant_normalized = (x_rounded / scale_factor) - self.z
        
        # 4. Rescale back (Multiply by gamma)
        x_out = x_quant_normalized * gamma

        return x_out

# QuantConv2d 和 QuantLinear 只需要微调初始化参数，去掉 num_channels，
# 因为 ReScaW 的 L1-mean 是对整个张量计算的（通常），或者对输出通道计算。
# 论文中公式 implied per-tensor or per-layer scaling usually for simple rescaling.
# 为了兼容您原本的调用结构，我们可以保持接口不变。

class QuantConv2d(nn.Conv2d):
    def __init__(
        self,
        bits: int,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
        self.bits = bits
        # 修改：不再需要传入 num_channels，因为 ReScaW 在 forward 中动态计算
        self.w_quantizer = WeightFakeQuantizer(bits) if bits < 32 else None

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
        bias=True
    ):
        super().__init__(in_features, out_features, bias)
        self.bits = bits
        self.w_quantizer = WeightFakeQuantizer(bits) if bits < 32 else None

    def extra_repr(self):
        return f"bits={self.bits}"

    def forward(self, input: torch.Tensor):
        w_fq = self.w_quantizer(self.weight) if self.bits < 32 else self.weight
        output = F.linear(input, w_fq, self.bias)
        return output