"""
This fusion algorithm is based on the paper "CM-UNet: Hybrid CNN-Mamba UNet for Remote Sensing Image Semantic Segmentation"
https://github.com/XiaoBuL/CM-UNet
Feature Fusion layer MSAA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ChannelAttentionModule1D(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ChannelAttentionModule1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_out = self.fc(self.avg_pool(x))  # (B, C, 1)
        max_out = self.fc(self.max_pool(x))  # (B, C, 1)
        out = avg_out + max_out
        return self.sigmoid(out)  # (B, C, 1)


class SpatialAttentionModule1D(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule1D, self).__init__()
        padding = kernel_size // 2
        # Conv1d输入维度是(B, C_in, L)，输出(B, 1, L)
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, L)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, L)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, L)
        x_out = self.conv1(x_cat)  # (B, 1, L)
        return self.sigmoid(x_out)


class FusionConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, factor=4.0):
        super(FusionConv1D, self).__init__()
        dim = int(out_channels // factor)
        self.down = nn.Conv1d(in_channels, dim, kernel_size=1, stride=1)
        self.conv_3 = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.conv_5 = nn.Conv1d(dim, dim, kernel_size=5, stride=1, padding=2)
        self.conv_7 = nn.Conv1d(dim, dim, kernel_size=7, stride=1, padding=3)

        self.spatial_attention = SpatialAttentionModule1D()
        self.channel_attention = ChannelAttentionModule1D(dim)

        self.up = nn.Conv1d(dim, out_channels, kernel_size=1, stride=1)

    def forward(self, Attentions):

        # 原始特征: Attentions (B, L, E)
        x_input = [rearrange(x, 'b e l -> b l e') for x in Attentions]  # (B, L, E) -> (B, E, L)

        x_fused = torch.cat(x_input, dim=1)  # (B, 3*E, L)
        x_fused = self.down(x_fused)  # (B, dim, L)

        x_fused_c = x_fused * self.channel_attention(x_fused)  # 通道注意力
        x_3 = self.conv_3(x_fused)
        x_5 = self.conv_5(x_fused)
        x_7 = self.conv_7(x_fused)

        x_fused_s = x_3 + x_5 + x_7  # (B, dim, L)
        x_fused_s = x_fused_s * self.spatial_attention(x_fused_s)  # 空间注意力(序列维度)

        x_out = self.up(x_fused_s + x_fused_c)  # (B, out_channels, L)

        # 输出若需要回到(B, L, E)形式再返回
        x_out = rearrange(x_out, 'b e l -> b l e')
        return x_out


class FeatureFusion1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusion1D, self).__init__()
        self.fusion_conv = FusionConv1D(in_channels, out_channels)

    def forward(self, Attentions):
        return self.fusion_conv(Attentions)

