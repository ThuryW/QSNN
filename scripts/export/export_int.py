import sys
import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
import pandas as pd
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# === 绘图库 ===
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==============================================================================
# 环境设置
# ==============================================================================
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    from engine.build import arch_dict, dataset_dict
except ImportError as e:
    print(f"[Error] 导入失败: {e}")
    sys.exit(1)

# ==============================================================================
# 全局数据容器
# ==============================================================================
EXPORT_DATA = {}     # 存储 Accum, Vmem, Input 等时序数据
LAYER_PARAMS = []    # 存储每一层的量化参数

def save_tensor_data(name, tensor, is_integer=True):
    """保存 Tensor 数据到全局字典 (仅保留 Batch 0)"""
    # tensor shape: [T, B, C, H, W] or [T, B, C]
    if tensor.shape[1] > 1:
        data = tensor[:, 0].detach().cpu().numpy() # 取 Batch 0
    else:
        # 如果 Batch 本身就是 1
        data = tensor.detach().cpu().numpy()
        if data.ndim > 1 and data.shape[1] == 1:
            data = data.squeeze(1) # [T, 1, ...] -> [T, ...]
        
    if is_integer:
        data = np.round(data).astype(int)
    
    EXPORT_DATA[name] = data

def register_layer_params(name, scale, bias, vth, tau, layer_type="Standard"):
    """记录硬件参数"""
    s = scale.detach().cpu().numpy()
    b = bias.detach().cpu().numpy()
    
    if isinstance(vth, torch.Tensor):
        v_val = vth.item() if vth.numel() == 1 else vth.mean().item()
    else:
        v_val = vth
        
    s_safe = s.copy()
    s_safe[s_safe == 0] = 1e-12
    
    # 计算理论上的硬件参数 K 和 Bias_Int
    param_k = np.round(tau / s_safe)
    param_bias = np.round(b / s_safe * tau)
    param_vth = np.round(v_val * param_k)
    
    channels = len(s)
    for c in range(channels):
        LAYER_PARAMS.append({
            'Layer': name,
            'Channel': c,
            'Type': layer_type,
            'Tau_Used': tau,
            'Scale_Float': s[c],
            'Bias_Float': b[c],
            'Vth_Float': v_val,
            'Param_K (Tau/Scale)': int(param_k[c]),
            'Param_Bias_Int': int(param_bias[c]),
            'Param_Vth_Int': int(param_vth[c])
        })

# ==============================================================================
# 1. 核心组件 (V4 Logic + Export Hooks)
# ==============================================================================

class IntegerNeuron(nn.Module):
    def __init__(self, timesteps, vth, reset_mode='soft', name="Neuron"):
        super().__init__()
        self.timesteps = timesteps
        self.reset_mode = reset_mode
        self.name = name
        self.register_buffer('vth', vth.clone().detach())

    def forward(self, x_accum, prev_scale, prev_bias, tau, is_first_layer=False):
        # --- V4 Logic Start: 维度对齐与参数计算 ---
        ndim = x_accum.ndim - 1 
        if ndim == 4: view_shape = (1, 1, -1, 1, 1)
        elif ndim == 2: view_shape = (1, 1, -1)
        else: view_shape = (1, 1) + (1,) * (ndim - 1)

        s_view = prev_scale.view(*view_shape)
        b_view = prev_bias.view(*view_shape)
        eps = 1e-12

        # 动态计算 Bias 和 Vth 的整数值
        bias_scaled_full = (b_view * tau / (s_view + eps)).round()
        vth_scaled_full = (self.vth * tau / (s_view + eps)).round()
        
        if bias_scaled_full.ndim == x_accum.ndim:
            bias_scaled = bias_scaled_full.squeeze(0)
            vth_scaled = vth_scaled_full.squeeze(0)
        else:
            bias_scaled = bias_scaled_full
            vth_scaled = vth_scaled_full

        # 准备驱动力
        if is_first_layer:
            current_drive_base = x_accum
        else:
            current_drive_base = x_accum * tau
        # --- V4 Logic End ---

        mem = torch.zeros_like(x_accum[0])
        spikes = []
        mem_history = [] # For Export

        for t in range(self.timesteps):
            if self.reset_mode == 'always_zero':
                mem.zero_()

            # 动力学方程
            mem = mem + current_drive_base[t] + bias_scaled
            
            spike = (mem >= vth_scaled).float()
            
            # Reset
            if self.reset_mode == 'soft':
                mem = mem - spike * vth_scaled
            elif self.reset_mode == 'hard':
                mem = mem * (1.0 - spike)
            elif self.reset_mode == 'always_zero':
                mem = mem - spike * vth_scaled
            
            spikes.append(spike)
            mem_history.append(mem.clone())

        # === Export Hook ===
        if self.name:
            save_tensor_data(f"{self.name}_Vmem", torch.stack(mem_history), is_integer=True)
            save_tensor_data(f"{self.name}_Spike", torch.stack(spikes), is_integer=True)
            
        return torch.stack(spikes, dim=0)


class IntegerConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, s, p, bias=False, name="Conv"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, k, s, p, bias=False)
        self.register_buffer('merged_scale', torch.ones(out_channels))
        self.register_buffer('merged_bias', torch.zeros(out_channels))
        self.name = name

    def forward(self, x):
        # 纯整数卷积累加
        T, B, C, H, W = x.shape
        x_reshaped = x.reshape(T * B, C, H, W)
        int_accum = self.conv(x_reshaped)
        
        # Reshape Back
        out = int_accum.reshape(T, B, *int_accum.shape[1:])
        
        # === Export Hook ===
        if self.name and B == 1:
            save_tensor_data(f"{self.name}_Accum", out, is_integer=True)
            
        return out


class IntegerLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False, name="Linear"):
        super().__init__()
        self.linear = nn.Linear(in_f, out_f, bias=False)
        self.register_buffer('merged_scale', torch.ones(out_f))
        self.register_buffer('merged_bias', torch.zeros(out_f))
        self.name = name

    def forward(self, x):
        T, B, C = x.shape
        x_reshaped = x.reshape(T * B, C)
        int_accum = self.linear(x_reshaped)
        out = int_accum.reshape(T, B, -1)
        
        # === Export Hook ===
        if self.name and B == 1:
            save_tensor_data(f"{self.name}_Accum", out, is_integer=True)
            
        return out

# ==============================================================================
# 2. Block Wrapper (V4 Logic: Fix Block0 Shortcut)
# ==============================================================================

class IntResBlockWrapper_Adv(nn.Module):
    def __init__(self, act1, conv1, act2, conv2, downsample):
        super().__init__()
        self.act1 = act1; self.conv1 = conv1
        self.act2 = act2; self.conv2 = conv2
        self.downsample = downsample

    def forward(self, x_accum, prev_scale, prev_bias, tau_act1, tau_act2, is_first_block_act1=False):
        # Act 1 (使用动态 Scale/Bias)
        s1 = self.act1(x_accum, prev_scale, prev_bias, tau_act1, is_first_layer=is_first_block_act1)
        out1 = self.conv1(s1)
        
        # Act 2
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
        
        # Scale 对齐
        eps = 1e-12
        ratio = s_short / (s_main + eps)
        
        # === 核心修正：消除 Block 0 Shortcut 的 tau_l0 膨胀 ===
        if is_shortcut_from_l0:
            ratio = ratio / tau_act1
            
        ratio_view = ratio.view(1, 1, -1, 1, 1)
        shortcut_aligned = (shortcut_accum * ratio_view).round()
        
        # Accumulator 与 Bias 相加
        final_accum = out2 + shortcut_aligned
        final_bias = b_main + b_short
        
        return final_accum, self.conv2.merged_scale, final_bias

# ==============================================================================
# 3. 转换与权重融合 (Adapted for ReScaW)
# ==============================================================================

def fuse_layer_weights(quant_conv, bn, prev_vth):
    """
    修改后的 Fusion 逻辑，适配 ReScaW 动态量化。
    """
    with torch.no_grad():
        w = quant_conv.weight.data
        
        # 1. 确定整数权重 (w_int) 和 缩放因子 (w_scale)
        if hasattr(quant_conv, 'w_quantizer') and quant_conv.w_quantizer is not None:
            # === ReScaW Logic ===
            bits = quant_conv.w_quantizer.bits
            s_b = 2 ** bits - 1
            
            # Recompute Gamma (L1-mean)
            gamma = w.abs().mean()
            if gamma == 0: gamma = 1e-8
            
            # 计算 ReScaW 原始索引: [0, s_b]
            w_normalized = (w / gamma).clamp(-1.0, 1.0)
            w_idx = ( (s_b / 2.0) * (w_normalized + 1.0) ).round()
            
            # 映射到对称整数域 (Symmetric Integer Domain)
            # e.g., 2-bit: 0->-3, 1->-1, 2->1, 3->3
            w_int_sym = 2.0 * w_idx - s_b
            
            # 计算对应的 Scale
            w_scale = gamma / s_b
            
            w_int_raw = w_int_sym
            w_scale_view = torch.tensor(w_scale, device=w.device).view(1)
            
        else:
            # === FP32/Standard Fallback (Layer 0) ===
            max_val = w.abs().max()
            if max_val == 0: max_val = 1.0
            w_scale = max_val / 127.0
            w_int_raw = (w / w_scale).round()
            w_scale_view = torch.tensor(w_scale, device=w.device).view(1)

        # 2. 处理 BN (Batch Normalization)
        if bn:
            mu, sigma = bn.running_mean, torch.sqrt(bn.running_var + bn.eps)
            gamma_bn, beta_bn = bn.weight, bn.bias
            bn_scale = gamma_bn / sigma
            bn_bias = beta_bn - (mu * bn_scale)
        else:
            bn_scale = torch.ones(w.shape[0], device=w.device) if w.dim() == 4 else torch.ones(w.shape[0], device=w.device)
            bn_bias = torch.zeros(w.shape[0], device=w.device)
            if quant_conv.bias is not None: 
                bn_bias = quant_conv.bias.data

        # 调整 scale 维度
        if w.dim() == 4:
            scale_combined = w_scale_view * bn_scale.view(-1, 1, 1, 1) * prev_vth
        else:
            scale_combined = w_scale_view * bn_scale.view(-1, 1) * prev_vth
            
        # 3. 融合 Scale 的符号到整数权重中
        scale_sign = torch.sign(scale_combined)
        scale_sign[scale_sign == 0] = 1.0 
        
        w_int_final = w_int_raw * scale_sign
        
        merged_bias = bn_bias
        merged_scale = scale_combined.abs().view(-1)
        
        return w_int_final, merged_scale, merged_bias

def convert_model(orig_model, device, tau_val, tau_l0_val, reset_mode='soft', l0_reset_mode='soft'):
    print(f">>> Converting Model for Export (V4 Logic + ReScaW, Tau={tau_val}, Tau_L0={tau_l0_val})...")
    
    # --- Layer 0 ---
    w, s, b = fuse_layer_weights(orig_model.conv1, orig_model.bn1, prev_vth=1.0)
    int_conv1 = IntegerConv2d(orig_model.conv1.in_channels, orig_model.conv1.out_channels, 3, 1, 1, name="Layer0_Stem") 
    if orig_model.conv1.kernel_size == 7: 
        int_conv1 = IntegerConv2d(orig_model.conv1.in_channels, orig_model.conv1.out_channels, 7, 2, 3, name="Layer0_Stem")
    
    int_conv1.conv.weight.data.copy_(w)
    int_conv1.merged_scale.data.copy_(s)
    int_conv1.merged_bias.data.copy_(b)
    
    # 记录参数
    first_block_vth = orig_model.layers[0].act1.vth.data
    register_layer_params("Layer0_Stem", s, b, first_block_vth, tau_l0_val, layer_type="Layer0")

    # --- Blocks ---
    int_blocks = []
    for i, layer in enumerate(orig_model.layers):
        blk_name = f"Block{i+1}"
        vth_in = layer.act1.vth.data
        
        i_act1 = IntegerNeuron(layer.act1.timesteps, vth_in, reset_mode=reset_mode, name=f"{blk_name}_Act1")
        
        w, s, b = fuse_layer_weights(layer.conv1, layer.bn1, vth_in)
        i_conv1 = IntegerConv2d(layer.conv1.in_channels, layer.conv1.out_channels, 3, layer.conv1.stride, 1, name=f"{blk_name}_Conv1")
        i_conv1.conv.weight.data.copy_(w); i_conv1.merged_scale.data.copy_(s); i_conv1.merged_bias.data.copy_(b)
        
        # Conv1 参数记录
        register_layer_params(f"{blk_name}_Conv1", s, b, layer.act2.vth.data, tau_val)
        
        vth2 = layer.act2.vth.data
        i_act2 = IntegerNeuron(layer.act2.timesteps, vth2, reset_mode=reset_mode, name=f"{blk_name}_Act2")
        
        w, s, b = fuse_layer_weights(layer.conv2, layer.bn2, vth2)
        i_conv2 = IntegerConv2d(layer.conv2.in_channels, layer.conv2.out_channels, 3, 1, 1, name=f"{blk_name}_Conv2")
        i_conv2.conv.weight.data.copy_(w); i_conv2.merged_scale.data.copy_(s); i_conv2.merged_bias.data.copy_(b)
        
        # Conv2 参数记录
        next_vth = orig_model.layers[i+1].act1.vth.data if i < len(orig_model.layers)-1 else orig_model.act1.vth.data
        register_layer_params(f"{blk_name}_Conv2", s, b, next_vth, tau_val)
        
        i_ds = None
        if layer.downsample:
            w, s, b = fuse_layer_weights(layer.downsample[0], layer.downsample[1], vth_in)
            ds = IntegerConv2d(layer.downsample[0].in_channels, layer.downsample[0].out_channels, 1, layer.downsample[0].stride, 0, name=f"{blk_name}_DS")
            ds.conv.weight.data.copy_(w); ds.merged_scale.data.copy_(s); ds.merged_bias.data.copy_(b)
            i_ds = nn.Sequential(ds)
            register_layer_params(f"{blk_name}_DS", s, b, next_vth, tau_val)
            
        int_blocks.append(IntResBlockWrapper_Adv(i_act1, i_conv1, i_act2, i_conv2, i_ds))

    # 第一层 Block 的 Act1 强制 soft reset
    if len(int_blocks) > 0:
        int_blocks[0].act1.reset_mode = l0_reset_mode

    # --- Final ---
    final_vth = orig_model.act1.vth.data
    final_act = IntegerNeuron(orig_model.act1.timesteps, final_vth, reset_mode=reset_mode, name="Final_Act")
    
    w, s, b = fuse_layer_weights(orig_model.fc, None, final_vth)
    int_fc = IntegerLinear(orig_model.fc.in_features, orig_model.fc.out_features, name="FC")
    int_fc.linear.weight.data.copy_(w); int_fc.merged_scale.data.copy_(s); int_fc.merged_bias.data.copy_(b)
    
    register_layer_params("FC", s, b, torch.tensor(0.0), tau_val)
    
    # --- Model Wrapper ---
    class IntegerResNetModel(nn.Module):
        def __init__(self, c1, blks, fact, fc, t):
            super().__init__()
            self.timesteps = t; self.conv1 = c1; self.blocks = nn.ModuleList(blks)
            self.final_act = fact; self.fc = fc; self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        def forward(self, x, tau, tau_l0):
            # Input Quantization
            x_input = (x * tau_l0).round()
            x_seq = x_input.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)
            save_tensor_data("SNN_Input_Int", x_seq, is_integer=True)
            
            # Layer 0
            l0_accum = self.conv1(x_seq)
            
            curr_accum = l0_accum
            curr_scale = self.conv1.merged_scale
            curr_bias = self.conv1.merged_bias
            
            # Blocks
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
            
            # Final
            spikes = self.final_act(curr_accum, curr_scale, curr_bias, tau=tau, is_first_layer=False)
            
            T, B, C, H, W = spikes.shape
            pooled = self.pool(spikes.reshape(T*B, C, H, W)).reshape(T, B, C)
            fc_accum = self.fc(pooled)
            
            # 导出 FC 最终结果
            scale_fc = self.fc.merged_scale.view(1, 1, -1)
            bias_fc = self.fc.merged_bias.view(1, 1, -1)
            logits = fc_accum * scale_fc + bias_fc
            
            return logits.mean(dim=0)

    return IntegerResNetModel(int_conv1, int_blocks, final_act, int_fc, orig_model.act1.timesteps).to(device)

# ==============================================================================
# 4. 辅助保存函数 (保持不变)
# ==============================================================================

def save_integer_weights_and_biases(model, out_dir, tau_val, tau_l0_val, timesteps):
    w_dir = os.path.join(out_dir, 'weights')
    os.makedirs(w_dir, exist_ok=True)
    l2_file_path = os.path.join(out_dir, 'weight_L2.txt')
    f_l2 = open(l2_file_path, 'w')
    global_max_info = {'val': -1.0, 'layer': None, 'channel': -1}
    layer_max_l2_map = {}
    
    for name, module in model.named_modules():
        weight = None
        if isinstance(module, (IntegerConv2d, nn.Conv2d)) and hasattr(module, 'conv'):
            weight = module.conv.weight.data
        elif isinstance(module, (IntegerLinear, nn.Linear)) and hasattr(module, 'linear'):
            weight = module.linear.weight.data
            
        if weight is not None and hasattr(module, 'name'):
            w_np = weight.detach().cpu().numpy().astype(int)
            w_flat = w_np.reshape(w_np.shape[0], -1) 
            np.savetxt(os.path.join(w_dir, f"{module.name}.csv"), w_flat, fmt='%d', delimiter=',')
            
            l2_norms = np.linalg.norm(w_flat, axis=1)
            current_layer_max = np.max(l2_norms)
            layer_max_l2_map[module.name] = current_layer_max
            
            current_max_idx = np.argmax(l2_norms)
            if l2_norms[current_max_idx] > global_max_info['val']:
                global_max_info['val'] = l2_norms[current_max_idx]
                global_max_info['layer'] = module.name
                global_max_info['channel'] = int(current_max_idx)
                
            for t in range(timesteps):
                l2_str = ', '.join([f"{val:.4f}" for val in l2_norms])
                f_l2.write(f"Layer: {module.name}, Timestep: {t+1}, Num_Filters: {len(l2_norms)}\n{l2_str}\n")
            
            # Bias 保存 (需要计算等效整数 Bias)
            if hasattr(module, 'merged_bias') and hasattr(module, 'merged_scale'):
                b = module.merged_bias
                s = module.merged_scale
                layer_tau = tau_l0_val if "Layer0" in module.name else tau_val
                # Bias_int = round(Bias * Tau / Scale)
                bias_int = (b / (s + 1e-12) * layer_tau).round()
                b_np = bias_int.detach().cpu().numpy().astype(int)
                np.savetxt(os.path.join(w_dir, f"{module.name}_bias.csv"), b_np, fmt='%d', delimiter=',')
                
    f_l2.close()
    return global_max_info, layer_max_l2_map

def save_summary_stats(out_dir, max_l2_info=None, layer_max_l2_map=None):
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w") as f:
        f.write(f"{'Layer Name':<40} | {'T':<3} | {'Type':<6} | {'Min':<12} | {'Max':<12} | {'Max_L2':<10}\n")
        f.write("-" * 100 + "\n")
        
        def sort_key(k):
            if "Input" in k: return "000_Input"
            if "Layer0" in k: return "001_" + k
            if "Final" in k or "FC" in k: return "zzz_" + k
            return k
            
        sorted_keys = sorted(EXPORT_DATA.keys(), key=sort_key)
        for name in sorted_keys:
            if "_Spike" in name: continue
            data = EXPORT_DATA[name] 
            data_type = name.split('_')[-1]
            num_timesteps = data.shape[0]
            
            base_layer_name = name.rsplit('_', 1)[0]
            l2_val_str = "-"
            if layer_max_l2_map and base_layer_name in layer_max_l2_map:
                l2_val_str = f"{layer_max_l2_map[base_layer_name]:.4f}"
            
            for t in range(num_timesteps):
                step_data = data[t]
                f.write(f"{name:<40} | {t+1:<3} | {data_type:<6} | {int(np.min(step_data)):<12} | {int(np.max(step_data)):<12} | {l2_val_str:<10}\n")
        
        if max_l2_info:
            f.write(f"\nGlobal Max L2 Norm: {max_l2_info['val']:.4f} @ {max_l2_info['layer']}\n")

def analyze_distributions(out_dir, target_layers):
    analysis_dir = os.path.join(out_dir, 'distribution_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"Generating Histograms for {target_layers} in {analysis_dir} ...")

    for name in target_layers:
        if name not in EXPORT_DATA: continue
        data_tensor = EXPORT_DATA[name]
        T = data_tensor.shape[0]
        for t in range(T):
            step_data = data_tensor[t].flatten()
            
            p1 = np.percentile(step_data, 1)
            p99 = np.percentile(step_data, 99)

            plt.figure(figsize=(10, 6))
            plt.hist(step_data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
            
            info_text = (f"Min: {np.min(step_data)}\n"
                         f"Max: {np.max(step_data)}\n"
                         f"Mean: {np.mean(step_data):.2f}\n"
                         f"Std: {np.std(step_data):.2f}\n"
                         f"1%: {p1:.2f}\n"
                         f"99%: {p99:.2f}")

            plt.gca().text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
                           fontsize=10, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            plt.title(f"Distribution: {name} - Timestep {t+1}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(os.path.join(analysis_dir, f"{name}_T{t+1}.png"))
            plt.close()

# ==============================================================================
# 主程序
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default='0')
    parser.add_argument('-cpt', '--checkpoint', required=True)
    parser.add_argument('-dd', '--data_dir', default='/home/wangtianyu/dataset')
    parser.add_argument('--out_dir', default='./export_results_v4_strict', help='Directory to save results')
    parser.add_argument('--tau', default=10.0, type=float)
    parser.add_argument('--tau_l0', default=1000.0, type=float)
    parser.add_argument('-a', '--arch', default='resnet20', type=str)
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-wb', '--weight_bits', default=4, type=int)
    parser.add_argument('-t', '--timesteps', default=4, type=int)
    parser.add_argument('--reset_mode', default='soft', type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Exporting Data (V4 Strict Dynamics + ReScaW) | Tau={args.tau}, Tau_L0={args.tau_l0}, Mode={args.reset_mode}")
    
    # 1. 构建原始模型
    orig_model = arch_dict(spiking=True, bits=args.weight_bits, timesteps=args.timesteps, arch_name=args.arch, dataset_name=args.dataset)
    
    # 2. 加载权重 (增加鲁棒性逻辑)
    print(f"Loading Checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
    
    # 处理 module. 前缀
    new_state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
    
    # 使用 strict=False 加载，避免 quantizer 等 buffer 缺失报错
    msg = orig_model.load_state_dict(new_state_dict, strict=False)
    print(f"Load Result: {msg}")
    
    orig_model.to(device).eval()
    
    # 3. 使用 V4 Logic 进行转换 (Default L0 Reset = soft)
    model = convert_model(orig_model, device, args.tau, args.tau_l0, reset_mode=args.reset_mode, l0_reset_mode='soft')
    
    # 4. 权重导出
    max_l2_info, layer_max_l2_map = save_integer_weights_and_biases(model, args.out_dir, args.tau, args.tau_l0, args.timesteps)
    
    # 5. 推理一次 (Image 0)
    _, test_set = dataset_dict(args.dataset, args.arch, args.data_dir)
    loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    
    print(f"Running Inference on Image 0...")
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            print(f"Input Shape: {x.shape}, Label: {y.item()}")
            model(x, tau=args.tau, tau_l0=args.tau_l0)
            break 

    # 6. 保存参数
    df_params = pd.DataFrame(LAYER_PARAMS)
    df_params.to_csv(os.path.join(args.out_dir, "tau_params.csv"), index=False)

    # 7. 保存 Layer Outputs
    print("Saving Layer Outputs to CSVs...")
    for t in range(args.timesteps):
        t_dir = os.path.join(args.out_dir, 'output', f'T{t+1}')
        os.makedirs(t_dir, exist_ok=True)
        for name, data in EXPORT_DATA.items():
            if data.shape[0] <= t: continue
            step_data = data[t].flatten()
            np.savetxt(os.path.join(t_dir, f"{name}.csv"), step_data, fmt='%d', delimiter=',')
            
    # Summary
    save_summary_stats(args.out_dir, max_l2_info, layer_max_l2_map)
    
    # 分布分析
    analyze_distributions(args.out_dir, target_layers=[
        "Block1_Act1_Vmem", 
        "Block1_Act2_Vmem", 
        "Block2_Act1_Vmem", 
        "Block3_Act1_Vmem"
    ])
    
    print("Done.")

if __name__ == "__main__":
    main()