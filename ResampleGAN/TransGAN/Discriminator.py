import torch
import torch.nn as nn
from einops import rearrange
from typing import Union, List
from ResampleGAN.TransGAN.Layer import Encoder
from ResampleGAN.TransGAN.EmbeddingLayer import EmbeddingLayer

class Discriminator(nn.Module):
    def __init__(self, num_layers: int,
                 dim_input: int,
                 dim_attention: int,
                 num_heads: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 attention_type: Union[str, List[str]] ='original',
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
        super(Discriminator, self).__init__()

        # Input embedding and projection
        self.input_embedding = EmbeddingLayer()
        self.input_proj = nn.Linear(dim_input, dim_attention)

        # Encoder-Decoder structure
        self.encoder = Encoder(num_layers, dim_attention, num_heads, dim_feedforward, dropout, attention_type, with_bias)

        # Output projection layers
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # 特征投影层 - 将编码器输出映射到特征空间
        self.feature_projector = nn.Sequential(
            nn.Linear(dim_attention, dim_attention * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_attention * 2, dim_attention),
            nn.LayerNorm(dim_attention)  # 特征归一化
        )

        self.classification_head = nn.Sequential(
            nn.Linear(dim_attention, dim_attention // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_attention // 2, 1)  # 输出一个单一的logit值
        )


    def forward(self, x_input, s_input, return_features = False):
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

        # 3. 全局池化
        #    为了使用1D池化，需要将维度从 (B, L, E) 变为 (B, E, L)
        features_permuted = rearrange(memory, 'b l e -> b e l')
        pooled_features = self.pooling(features_permuted)
        # pooled_features = rearrange(pooled_features, 'b e l -> b e')
        pooled_features = pooled_features.squeeze(-1)
        features = self.feature_projector(pooled_features)

        # classification head
        logits = self.classification_head(features)

        if return_features:
            return logits, features
        return logits

