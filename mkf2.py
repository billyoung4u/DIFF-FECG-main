#改进后的去噪网络

# mkf.py —— 仅按 Conv1dBlock 规范改“单卷积”，多核分支 ks 保留

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log as ln

# 采用与 unet1d 相同的 Conv1dBlock：Conv1d + GroupNorm(n_groups=8) + Mish，padding = k//2
try:
    from .conv1d_components import Conv1dBlock
except Exception:
    from conv1d_components import Conv1dBlock

Linear = nn.Linear

class Conv1d(nn.Conv1d):
    """保留原始裸卷积（多核块内部仍用），Kaiming 初始化"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class PositionalEncoding(nn.Module):
    """与原 mkf 保持一致：64 维正弦时间嵌入（不改此处，以避免非卷积因素影响对比）"""
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim
    def forward(self, noise_level):
        noise_level = noise_level.view(-1)
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype, device=noise_level.device) / count
        encoding = noise_level.unsqueeze(1) * torch.exp(-ln(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding  # [B,64]

class FeatureWiseAffine(nn.Module):
    """与原 mkf 一致的 FiLM 调制"""
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super().__init__()
        self.use_affine_level = use_affine_level
        self.diffusion_projection = Linear(64, in_channels)
        self.noise_func = nn.Linear(in_channels, out_channels * (1 + int(use_affine_level)))
    def forward(self, x, noise_embed):
        b = x.shape[0]
        h = self.diffusion_projection(noise_embed)  # [B, Cin]
        h = h.expand(b, -1)
        if self.use_affine_level:
            gamma, beta = self.noise_func(h).view(b, -1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(h).view(b, -1, 1)
        return x

# ---------------- 多核卷积块（kernel size 按原版保留：3/5/9/15） ----------------
class HNFBlock(nn.Module):
    """
    仅校正 padding 规则为 same-length（padding=(k//2)*dilation），
    其余逻辑保留（包含 chunk、InstanceNorm、LeakyReLU）。
    """
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        pad_mode = 'zeros'
        self.filters = nn.ModuleList([
            Conv1d(input_size, hidden_size // 4, 3,  dilation=dilation, padding=1 * dilation, padding_mode=pad_mode),
            Conv1d(input_size, hidden_size // 4, 5,  dilation=dilation, padding=2 * dilation, padding_mode=pad_mode),
            Conv1d(input_size, hidden_size // 4, 9,  dilation=dilation, padding=4 * dilation, padding_mode=pad_mode),
            Conv1d(input_size, hidden_size // 4, 15, dilation=dilation, padding=7 * dilation, padding_mode=pad_mode),
        ])
        # 聚合后两层仍用裸卷积，只按 Conv1dBlock 的 k=9, pad=4 规范对齐
        self.conv_1 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode=pad_mode)
        self.norm   = nn.InstanceNorm1d(hidden_size // 2)  # 保留
        self.conv_2 = Conv1d(hidden_size, hidden_size, 9, padding=4, padding_mode=pad_mode)

    def forward(self, x):
        residual = x
        filts = torch.cat([layer(x) for layer in self.filters], dim=1)
        nfilts, filts = self.conv_1(filts).chunk(2, dim=1)
        filts = F.leaky_relu(torch.cat([self.norm(nfilts), filts], dim=1), 0.5)
        filts = F.leaky_relu(self.conv_2(filts), 0.5)
        return filts + residual

# ---------------- 条件门控块（对齐 Conv1dBlock 的卷积参数/规范） ----------------
class DiffBlock(nn.Module):
    """
    门控的扩张卷积采用 k=9、padding=(k//2)*dilation 的 same-length 规范；
    在 1x1 输出后补一个 Conv1dBlock(hidden, hidden, k=9) 做 GN+Mish（不改变结构）。
    """
    def __init__(self, input_size, hidden_size, dilation):
        super().__init__()
        k, p = 9, (9 // 2) * dilation
        self.dilated_conv   = Conv1d(hidden_size, 2 * hidden_size, k, padding=p, dilation=dilation)
        self.output_residual= Conv1d(hidden_size, hidden_size, 1)
        self.post_norm_act  = Conv1dBlock(hidden_size, hidden_size, kernel_size=9)

    def forward(self, x, con):
        y = self.dilated_conv(con)
        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = y + x
        y = self.output_residual(y)
        y = self.post_norm_act(y)
        return y

# ---------------- Bridge：两侧卷积替换为 Conv1dBlock（k=9, pad=4） ----------------
class Bridge(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_conv  = Conv1dBlock(input_size,  input_size,  kernel_size=9)
        self.encoding    = FeatureWiseAffine(input_size, hidden_size, use_affine_level=1)
        self.output_conv = Conv1dBlock(input_size,  hidden_size, kernel_size=9)
    def forward(self, x, noise_embed):
        x = self.input_conv(x)
        x = self.encoding(x, noise_embed)
        x = self.output_conv(x)
        return x

# ---------------- 顶层：仅把“单卷积”位置替换为 Conv1dBlock ----------------
class ConditionalModel(nn.Module):
    def __init__(self, feats=64):   # 与原 mkf 默认一致
        super(ConditionalModel, self).__init__()
        # x-stream：首层 Conv 改用 Conv1dBlock(k=9)
        self.stream_x = nn.ModuleList([
            nn.Sequential(Conv1dBlock(1, feats, kernel_size=9)),
            HNFBlock(feats, feats, 1),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 4),
            HNFBlock(feats, feats, 2),
            HNFBlock(feats, feats, 1),
        ])
        # cond-stream：首层 Conv 改用 Conv1dBlock(k=9)
        self.stream_cond = nn.ModuleList([
            nn.Sequential(Conv1dBlock(1, feats, kernel_size=9)),
            DiffBlock(feats, feats, 1),
            DiffBlock(feats, feats, 2),
            DiffBlock(feats, feats, 4),
            DiffBlock(feats, feats, 2),
            DiffBlock(feats, feats, 1),
        ])

        self.embed = PositionalEncoding(64)

        # Bridge 个数与 x-stream 层数对齐（6 个），避免 zip 截断
        self.bridge = nn.ModuleList([Bridge(feats, feats) for _ in range(len(self.stream_x))])

        # 输出头：参考 unet1d，先 k=9 的 Conv1dBlock，再 1x1
        self.conv_out = nn.Sequential(
            Conv1dBlock(feats, feats, kernel_size=9),
            nn.Conv1d(feats, 1, 1),
        )

    def forward(self, x, cond, noise_scale):
        x = x.unsqueeze(1)
        cond = cond.unsqueeze(1)
        noise_embed = self.embed(noise_scale)

        xs = []
        for layer, br in zip(self.stream_x, self.bridge):
            x = layer(x)
            xs.append(br(x, noise_embed))

        for i, (xb, layer) in enumerate(zip(xs, self.stream_cond)):
            if i == 0:
                cond = layer(cond) + xb
            else:
                cond = layer(xb, cond)

        return self.conv_out(cond)
