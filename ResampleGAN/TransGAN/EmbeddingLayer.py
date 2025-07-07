import torch
import torch.nn as nn
import pandas as pd
import numpy as np

class EmbeddingLayer(nn.Module):
    def __init__(self):
        super(EmbeddingLayer, self).__init__()

    def forward(self, x_input, s_input):
        """
        前向传播
        :param x_input: 输入张量, 形状为 (batch_size, len_input, d)
        :param s_input: 时间分辨率tuple, 每个元素为字符串(如 '5min')
        :return: x: 输出张量, 形状为 (batch_size, len_input, d)
        """
        batch_size, len_input, d_input = x_input.shape

        # 计算每个样本的时间序列位置信息
        positions = self.calculate_positions(batch_size, len_input, s_input, x_input.device)
        # 生成位置编码
        pe = self.positional_encoding(batch_size, len_input, d_input, positions, x_input.device)
        # 将位置编码加到输入序列
        x = x_input + pe
        return x

    def calculate_positions(self, batch_size, len_input, s_input, device='cpu'):
        """
        计算时间序列的实际位置索引

        :param batch_size: 批次大小
        :param len_input: 序列长度
        :param s_input: 时间分辨率tuple，每个元素为字符串
        :param device: 设备
        :return: positions: 实际位置索引, 形状为 (batch_size, len_input)
        """
        # 创建基础位置索引 [len_input]
        base_positions = torch.arange(0, len_input, device=device).float()

        # 解析每个样本的时间分辨率并转换为tensor
        freq_deltas = []
        for s in s_input:
            minutes = pd.Timedelta(s).total_seconds() / 60
            freq_deltas.append(minutes)

        # 转换为tensor并扩展维度
        freq_deltas = torch.tensor(freq_deltas, device=device).float().view(-1, 1)

        # 将freq_deltas扩展为 (batch_size, len_input)
        freq_deltas = freq_deltas.expand(-1, len_input)

        # 计算每个样本的实际位置 (batch_size, len_input)
        positions = base_positions.unsqueeze(0) * freq_deltas
        return positions

    def positional_encoding(self, batch_size, len_input, d_input, positions, device='cpu'):
        """
        生成正弦余弦位置编码

        :param batch_size: 批次大小
        :param len_input: 序列长度
        :param d_input: 位置编码维度
        :param positions: 实际位置索引, 形状为 (batch_size, len_input)
        :param device: 设备
        :return: position_encoding: 位置编码张量, 形状为 (batch_size, len_input, d_input)
        """
        # 生成维度序列 [d//2]
        div_term = torch.exp(
            torch.arange(0, d_input, 2, device=device).float() *
            (-np.log(10000.0) / d_input)
        )

        # 创建空的位置编码矩阵 [batch_size, len_input, d_input]
        pe = torch.zeros(batch_size, len_input, d_input, device=device)

        # 分别计算sin和cos位置编码
        # positions现在是 (batch_size, len_input)，需要增加一个维度来进行广播
        positions_expanded = positions.unsqueeze(-1)

        pe[:, :, 0::2] = torch.sin(positions_expanded * div_term)
        pe[:, :, 1::2] = torch.cos(positions_expanded * div_term)

        return pe