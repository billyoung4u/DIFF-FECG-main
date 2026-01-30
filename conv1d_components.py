import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """Mish激活函数，通常比ReLU效果更好"""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Conv1dBlock(nn.Module):
    """
    mkf2.py 依赖的基础卷积块：
    Conv1d -> GroupNorm -> Mish
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None, n_groups=8):
        super().__init__()
        # 自动计算 padding 保持 Same Padding
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

        # GroupNorm 比 BatchNormalization 在小 batch size 下更稳定
        # 确保通道数能被 n_groups 整除，如果不行就设为 1 (LayerNorm) 或其他因数
        if out_channels % n_groups != 0:
            # 简单的容错处理，防止报错
            n_groups = 1

        self.norm = nn.GroupNorm(n_groups, out_channels)
        self.act = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x