import sys
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from datetime import datetime

# ==============================================================================
# 环境与日志设置
# ==============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    from engine.build import arch_dict, dataset_dict
except ImportError as e:
    print(f"[Error] 导入失败: {e}")
    sys.exit(1)

logger = None
def setup_logger(output_dir):
    global logger
    os.makedirs(output_dir, exist_ok=True)
    log_name = f"inspect_int_dynamics_rescaw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(output_dir, log_name)
    logger = logging.getLogger("Inspect")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers(): logger.handlers.clear()
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter); ch.setFormatter(formatter)
    logger.addHandler(fh); logger.addHandler(ch)
    return log_path

def log_print(msg):
    if logger: logger.info(msg)
    else: print(msg)

# ==============================================================================
# 1. 核心组件 (保持 V3 的逻辑)
# ==============================================================================

class IntegerNeuron(nn.Module):
    def __init__(self, timesteps, vth, reset_mode='soft', layer_name='neuron'):
        super().__init__()
        self.timesteps = timesteps
        self.reset_mode = reset_mode
        self.layer_name = layer_name
        self.register_buffer('vth', vth.clone().detach())

    def forward(self, x_accum, prev_scale, prev_bias, tau, is_first_layer=False):
        ndim = x_accum.ndim - 1 
        if ndim == 4: view_shape = (1, 1, -1, 1, 1)
        elif ndim == 2: view_shape = (1, 1, -1)
        else: view_shape = (1, 1) + (1,) * (ndim - 1)

        s_view = prev_scale.view(*view_shape)
        b_view = prev_bias.view(*view_shape)
        eps = 1e-12

        bias_scaled_full = (b_view * tau / (s_view + eps)).round()
        vth_scaled_full = (self.vth * tau / (s_view + eps)).round()
        
        if bias_scaled_full.ndim == x_accum.ndim:
            bias_scaled = bias_scaled_full.squeeze(0)
            vth_scaled = vth_scaled_full.squeeze(0)
        else:
            bias_scaled = bias_scaled_full
            vth_scaled = vth_scaled_full

        if is_first_layer:
            current_drive_base = x_accum
        else:
            current_drive_base = x_accum * tau

        mem = torch.zeros_like(x_accum[0])
        spikes = []

        for t in range(self.timesteps):
            if self.reset_mode == 'always_zero':
                mem.zero_()

            mem = mem + current_drive_base[t] + bias_scaled
            spike = (mem >= vth_scaled).float()
            spikes.append(spike)

            if self.reset_mode == 'soft':
                mem = mem - spike * vth_scaled
            elif self.reset_mode == 'hard':
                mem = mem * (1.0 - spike)
            elif self.reset_mode == 'always_zero':
                mem = mem - spike * vth_scaled

        return torch.stack(spikes, dim=0)


class IntegerConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.register_buffer('merged_scale', torch.ones(out_channels))
        self.register_buffer('merged_bias', torch.zeros(out_channels))

    def forward(self, x):
        T, B, C, H, W = x.shape
        x_reshaped = x.reshape(T * B, C, H, W)
        int_accum = self.conv(x_reshaped)
        return int_accum.reshape(T, B, *int_accum.shape[1:])

class IntegerLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.register_buffer('merged_scale', torch.ones(out_f))
        self.register_buffer('merged_bias', torch.zeros(out_f))

    def forward(self, x):
        T, B, C = x.shape
        x_reshaped = x.reshape(T * B, C)
        int_accum = self.linear(x_reshaped)
        return int_accum.reshape(T, B, -1)

# ==============================================================================
# 2. Block Wrapper (Block0 Shortcut 归一化)
# ==============================================================================

class IntResBlockWrapper_Adv(nn.Module):
    def __init__(self, act1, conv1, act2, conv2, downsample):
        super().__init__()
        self.act1 = act1; self.conv1 = conv1
        self.act2 = act2; self.conv2 = conv2
        self.downsample = downsample

    def forward(self, x_accum, prev_scale, prev_bias, tau_act1, tau_act2, is_first_block_act1=False):
        # Act 1 & Conv 1
        s1 = self.act1(x_accum, prev_scale, prev_bias, tau_act1, is_first_layer=is_first_block_act1)
        out1 = self.conv1(s1)
        
        # Act 2 & Conv 2
        s2 = self.act2(out1, self.conv1.merged_scale, self.conv1.merged_bias, tau_act2, is_first_layer=False)
        out2 = self.conv2(s2)
        
        # Shortcut 处理
        is_shortcut_from_l0 = False
        if self.downsample:
            shortcut_accum = self.downsample[0](s1)
            s_short = self.downsample[0].merged_scale
            b_short = self.downsample[0].merged_bias
        else:
            shortcut_accum = x_accum
            s_short = prev_scale
            b_short = prev_bias
            if is_first_block_act1:
                is_shortcut_from_l0 = True
        
        s_main = self.conv2.merged_scale
        b_main = self.conv2.merged_bias
        
        # Scale Alignment
        eps = 1e-12
        ratio = s_short / (s_main + eps)
        
        if is_shortcut_from_l0:
            ratio = ratio / tau_act1
            
        ratio_view = ratio.view(1, 1, -1, 1, 1)
        shortcut_aligned = (shortcut_accum * ratio_view).round()
        
        final_accum = out2 + shortcut_aligned
        final_bias = b_main + b_short
        
        return final_accum, self.conv2.merged_scale, final_bias

# ==============================================================================
# 3. 转换逻辑 (ADAPTED FOR QP-SNN ReScaW)
# ==============================================================================

def fuse_layer_weights(quant_conv, bn, prev_vth):
    """
    修改后的 Fusion 逻辑，适配 ReScaW 动态量化。
    ReScaW (2-bit) 产生整数 {0, 1, 2, 3}，这等价于对称整数 {-3, -1, 1, 3} 配合特定的 scale。
    这里我们将 ReScaW 的参数转换为对称整数形式，以便 IntegerConv2d 可以正确处理。
    """
    with torch.no_grad():
        w = quant_conv.weight.data
        
        # 1. 确定整数权重 (w_int) 和 缩放因子 (w_scale)
        if hasattr(quant_conv, 'w_quantizer') and quant_conv.w_quantizer is not None:
            # === ReScaW 逻辑 ===
            bits = quant_conv.w_quantizer.bits
            s_b = 2 ** bits - 1
            
            # Recompute Gamma (L1-mean) 这里的计算必须与 quant_layers.py 一致
            gamma = w.abs().mean()
            if gamma == 0: gamma = 1e-8
            
            # 计算 ReScaW 原始索引: [0, s_b]
            # Formula: round( s(b)/2 * (clamp(w/gamma, -1, 1) + 1) )
            # 注意：z=1, scale_factor = s_b/2
            w_normalized = (w / gamma).clamp(-1.0, 1.0)
            w_idx = ( (s_b / 2.0) * (w_normalized + 1.0) ).round()
            
            # 映射到对称整数域 (Symmetric Integer Domain)
            # 2-bit: 0->-3, 1->-1, 2->1, 3->3. Formula: 2*idx - s_b
            w_int_sym = 2.0 * w_idx - s_b
            
            # 计算对应的 Scale
            # Formula: w_float ~= w_int_sym * (gamma / s_b)
            w_scale = gamma / s_b
            
            w_int_raw = w_int_sym
            # 将标量 scale 转为 tensor 方便后续广播计算
            w_scale_view = torch.tensor(w_scale, device=w.device).view(1)
            
        else:
            # === FP32 Layer Fallback (通常第一层) ===
            # 使用简单的 Max-Scaling 映射到 8-bit 整数范围，保证精度
            max_val = w.abs().max()
            if max_val == 0: max_val = 1.0
            w_scale = max_val / 127.0
            w_int_raw = (w / w_scale).round()
            w_scale_view = torch.tensor(w_scale, device=w.device).view(1)

        # 2. 处理 BN (Batch Normalization)
        if bn:
            mu, sigma = bn.running_mean, torch.sqrt(bn.running_var + bn.eps)
            gamma_bn, beta_bn = bn.weight, bn.bias
            
            # BN 的等效 Scale 和 Bias
            bn_scale = gamma_bn / sigma
            bn_bias = beta_bn - (mu * bn_scale)
        else:
            bn_scale = torch.ones(w.shape[0], device=w.device) if w.dim() == 4 else torch.ones(w.shape[0], device=w.device)
            bn_bias = torch.zeros(w.shape[0], device=w.device)
            if quant_conv.bias is not None: 
                bn_bias = quant_conv.bias.data

        # 调整 bn_scale 的维度以匹配 w_scale
        if w.dim() == 4:
            # Conv: w_scale 是标量, bn_scale 是 (C_out)
            # 我们需要 total_scale 为 (C_out, 1, 1, 1) 
            scale_combined = w_scale_view * bn_scale.view(-1, 1, 1, 1) * prev_vth
        else:
            # Linear: w_scale 是标量, bn_scale 是 (Out)
            scale_combined = w_scale_view * bn_scale.view(-1, 1) * prev_vth
            
        # 3. 融合 Scale 的符号到整数权重中
        # IntegerConv2d 执行的是 w_int * x。我们把总符号 (sign) 并在 w_int 里。
        # 绝对值 (abs) 留在 merged_scale 里供下一层 neuron 处理。
        scale_sign = torch.sign(scale_combined)
        scale_sign[scale_sign == 0] = 1.0 # 避免 0 符号
        
        w_int_final = w_int_raw * scale_sign
        
        merged_bias = bn_bias
        merged_scale = scale_combined.abs().view(-1) # Flatten for buffer storage
        
        return w_int_final, merged_scale, merged_bias

def convert_model(orig_model, device, dataset_name):
    log_print(f">>> Converting Model (QP-SNN ReScaW Adapted)...")
    
    # Layer 0
    w, s, b = fuse_layer_weights(orig_model.conv1, orig_model.bn1, prev_vth=1.0)
    int_conv1 = IntegerConv2d(orig_model.conv1.in_channels, orig_model.conv1.out_channels, 3, 1, 1) 
    if orig_model.conv1.kernel_size == 7: 
        int_conv1 = IntegerConv2d(orig_model.conv1.in_channels, orig_model.conv1.out_channels, 7, 2, 3)
    
    int_conv1.conv.weight.data.copy_(w)
    int_conv1.merged_scale.data.copy_(s)
    int_conv1.merged_bias.data.copy_(b)
    
    # Blocks
    int_blocks = []
    for layer in orig_model.layers:
        vth_in = layer.act1.vth.data
        i_act1 = IntegerNeuron(layer.act1.timesteps, vth_in, layer_name="act1")
        
        w, s, b = fuse_layer_weights(layer.conv1, layer.bn1, vth_in)
        i_conv1 = IntegerConv2d(layer.conv1.in_channels, layer.conv1.out_channels, 3, layer.conv1.stride, 1)
        i_conv1.conv.weight.data.copy_(w); i_conv1.merged_scale.data.copy_(s); i_conv1.merged_bias.data.copy_(b)
        
        vth2 = layer.act2.vth.data
        i_act2 = IntegerNeuron(layer.act2.timesteps, vth2, layer_name="act2")
        
        w, s, b = fuse_layer_weights(layer.conv2, layer.bn2, vth2)
        i_conv2 = IntegerConv2d(layer.conv2.in_channels, layer.conv2.out_channels, 3, 1, 1)
        i_conv2.conv.weight.data.copy_(w); i_conv2.merged_scale.data.copy_(s); i_conv2.merged_bias.data.copy_(b)
        
        i_ds = None
        if layer.downsample:
            # 注意：Downsample 层通常也是 Conv，也需要应用 fuse_layer_weights
            w, s, b = fuse_layer_weights(layer.downsample[0], layer.downsample[1], vth_in)
            ds = IntegerConv2d(layer.downsample[0].in_channels, layer.downsample[0].out_channels, 1, layer.downsample[0].stride, 0)
            ds.conv.weight.data.copy_(w); ds.merged_scale.data.copy_(s); ds.merged_bias.data.copy_(b)
            i_ds = nn.Sequential(ds)
            
        int_blocks.append(IntResBlockWrapper_Adv(i_act1, i_conv1, i_act2, i_conv2, i_ds))

    # 第一层 soft reset
    if len(int_blocks) > 0:
        int_blocks[0].act1.reset_mode = 'soft'

    # Final
    final_vth = orig_model.act1.vth.data
    final_act = IntegerNeuron(orig_model.act1.timesteps, final_vth, layer_name="final_act")
    
    w, s, b = fuse_layer_weights(orig_model.fc, None, final_vth)
    int_fc = IntegerLinear(orig_model.fc.in_features, orig_model.fc.out_features)
    int_fc.linear.weight.data.copy_(w); int_fc.merged_scale.data.copy_(s); int_fc.merged_bias.data.copy_(b)
    
    class IntegerResNetModel(nn.Module):
        def __init__(self, c1, blks, fact, fc, t):
            super().__init__()
            self.timesteps = t; self.conv1 = c1; self.blocks = nn.ModuleList(blks)
            self.final_act = fact; self.fc = fc; self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        def forward(self, x, tau, tau_l0):
            # Input Quantization (Input is usually 8-bit image 0-255 or normalized float)
            # Here we assume x is normalized float [0,1] or similar.
            # Convert to integer drive for first layer.
            x_input = (x * tau_l0).round()
            x_seq = x_input.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)
            
            l0_accum = self.conv1(x_seq)
            
            curr_accum = l0_accum
            curr_scale = self.conv1.merged_scale
            curr_bias = self.conv1.merged_bias
            
            for i, block in enumerate(self.blocks):
                is_first = (i == 0)
                t_a1 = tau_l0 if is_first else tau
                t_a2 = tau
                
                curr_accum, curr_scale, curr_bias = block(
                    curr_accum, curr_scale, curr_bias, 
                    tau_act1=t_a1, 
                    tau_act2=t_a2, 
                    is_first_block_act1=is_first
                )
            
            spikes = self.final_act(curr_accum, curr_scale, curr_bias, tau=tau, is_first_layer=False)
            
            T, B, C, H, W = spikes.shape
            pooled = self.pool(spikes.reshape(T*B, C, H, W)).reshape(T, B, C)
            fc_accum = self.fc(pooled)
            
            scale_fc = self.fc.merged_scale.view(1, 1, -1)
            bias_fc = self.fc.merged_bias.view(1, 1, -1)
            logits = fc_accum * scale_fc + bias_fc
            
            return logits.mean(dim=0)

    return IntegerResNetModel(int_conv1, int_blocks, final_act, int_fc, orig_model.act1.timesteps).to(device)

def get_standard_loader(dataset_name, arch_name, data_dir, batch_size):
    # 这里需要确保 engine.build 能够正确返回 dataset
    _, test_set = dataset_dict(dataset_name, arch_name, data_dir)
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default='0', type=str)
    parser.add_argument('-cpt', '--checkpoint', type=str, required=True)
    parser.add_argument('-dd', '--data_dir', default='/home/wangtianyu/dataset', type=str)
    parser.add_argument('-a', '--arch', default='resnet20', type=str)
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-t', '--timesteps', default=4, type=int)
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('--tau', default=10.0, type=float, help="Neuron scaling factor, e.g., 2**10 or 1000")
    parser.add_argument('--tau_l0', default=1000.0, type=float, help="Input layer scaling factor")
    parser.add_argument('--log_dir', default='./logs/inspect', type=str)
    parser.add_argument('-wb', '--weight_bits', default=2, type=int, help="Bits used in training (e.g. 2)")
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_logger(args.log_dir)
    
    log_print(f"SNN Strict Integer Dynamics (ReScaW Adapted) | Bits: {args.weight_bits} | Timesteps: {args.timesteps}")
    
    # 构建原始模型结构 (Float/FakeQuant)
    orig_model = arch_dict(spiking=True, bits=args.weight_bits, timesteps=args.timesteps, arch_name=args.arch, dataset_name=args.dataset)
    
    # 加载权重
    log_print(f"Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    
    # 处理 module. 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') 
        new_state_dict[name] = v
        
    msg = orig_model.load_state_dict(new_state_dict, strict=False)
    log_print(f"Load result: {msg}")
    
    orig_model.to(device).eval()
    
    # 执行整数转换
    int_model = convert_model(orig_model, device, args.dataset)
    
    # 获取数据加载器
    loader = get_standard_loader(args.dataset, args.arch, args.data_dir, args.batch_size)
    
    correct=0; total=0
    log_print("Starting Integer Inference...")
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device); y = y.to(device)
            # 运行整数模型
            out = int_model(x, tau=args.tau, tau_l0=args.tau_l0)
            
            correct += out.argmax(1).eq(y).sum().item(); total += y.size(0)
            if i % 10 == 0: log_print(f"Batch {i}/{len(loader)} done. Acc: {100.*correct/total:.2f}%")
            
    acc = 100.*correct/total
    log_print(f"Final Integer Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    main()