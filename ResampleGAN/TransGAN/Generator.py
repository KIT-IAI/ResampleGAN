import torch
import torch.nn as nn
from typing import Union, List
from ResampleGAN.TransGAN.Layer import Encoder, Decoder
from ResampleGAN.TransGAN.EmbeddingLayer import EmbeddingLayer

class Generator(nn.Module):
    def __init__(self, num_layers: int,
                 dim_input: int,
                 dim_attention: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 use_noise: bool = False,
                 noise_scale: float = 0.1,
                 use_mask: bool = False,
                 attention_type: Union[str, List[str]] ='original',
                 restore: bool = True,
                 with_bias: bool = False):
        """
        生成器模型
        :param num_layers: 编码层和解码层的数量
        :param dim_input: 输入x_input维度
        :param dim_attention: 多头注意力机制的维度
        :param num_heads: 多头注意力机制的头数
        :param dim_feedforward: FNN的隐藏层维度
        :param dropout: dropout概率
        :param use_noise: 是否使用噪声（防止可能的模式崩溃）
        :param noise_scale: 噪声的标准差
        :param use_mask: 是否使用掩码
        :param attention_type: 注意力机制类型
        :param restore: 如果是插值且是否恢复原始值
        """
        super(Generator, self).__init__()
        # 噪声参数
        self.use_noise = use_noise
        self.noise_scale = noise_scale

        # 是否恢复原始值
        self.restore = restore

        # Input embedding and projection
        self.input_embedding = EmbeddingLayer()
        self.input_proj = nn.Linear(dim_input, dim_attention)

        # Output embedding and projection
        self.output_embedding = EmbeddingLayer()
        self.output_proj = nn.Linear(dim_input, dim_attention)

        # Encoder-Decoder structure
        self.encoder = Encoder(num_layers, dim_attention, num_heads, dim_feedforward, dropout, attention_type, with_bias)
        self.decoder = Decoder(num_layers, dim_attention, num_heads, dim_feedforward, dropout, use_mask, attention_type, with_bias)

        # Output projection layers
        self.output_hidden = nn.Linear(dim_attention, dim_attention // 2)
        self.output_final = nn.Linear(dim_attention // 2, dim_input)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def _parse_interval(self, interval_str):
        if 'min' in interval_str:
            return int(interval_str.replace('min', ''))
        elif 'H' in interval_str or 'h' in interval_str:
            return int(interval_str.replace('H', '').replace('h', '')) * 60
        else:
            raise ValueError(f"Unsupported interval format: {interval_str}")

    def forward(self, x_input, x_initial, s_input, s_output, mask=None, x_initial_mask=None, with_clamp=True):
        """
        前向传播
        :param x_input: 需要被插值/聚合的原始时间序列
        :param x_output: 和目标时间序列相同长度的时间序列
        :param s_input: 原始时间序列的分辨率
        :param s_output: 目标时间序列的分辨率
        :param mask: 因果掩码
        :return: output: 生成的时间序列
        """
        # Input embedding with optional noise
        x = self.input_embedding(x_input, s_input)
        x = self.input_proj(x)

        # Encoder
        memory = self.encoder(x)

        # Target embedding
        tgt = self.output_embedding(x_initial, s_output)
        tgt = self.output_proj(tgt)

        # Decoder
        decoded = self.decoder(tgt, memory, mask)

        # Output projection
        output = self.output_hidden(decoded)
        output = self.gelu(output)
        output = self.dropout(output)
        output = self.output_final(output)

        if x_initial_mask is not None:
            output[x_initial_mask == 1.] = x_initial[x_initial_mask == 1.]

        if with_clamp:
            output = output.clamp(min=-1, max=1)

        return output