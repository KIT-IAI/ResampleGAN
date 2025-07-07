"""
GAN模型的生成器, 使用了Transformer的Encoder-Decoder结构。
"""
import torch
import torch.nn as nn
from typing import Union, List, Optional

from ResampleGAN.TransGAN.Attention.AttentionRegistry import AttentionRegistry
from ResampleGAN.TransGAN.Attention.AttentionFusion import FeatureFusion1D

class EncodingLayer(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dim_feedforward: int,
                 dropout: float, attention_type: str, with_bias: bool):
        super().__init__()
        # 自注意力层
        self.self_attn, _ = AttentionRegistry.get_attention(attention_type, dim_attention, num_heads, dropout, with_bias)

        # 模块化前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_attention, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_attention)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim_attention)
        self.norm2 = nn.LayerNorm(dim_attention)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # 通道注意力
        # self.channel_attn = ChannelAttention1D(dim_attention)
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param src: source tensor from input or previous encodinglayer
        :return: src: encoded tensor
        """
        # Pre-norm architecture for self attention
        src = src + self.dropout1(self.self_attn(self.norm1(src)))
        src = src + self.dropout2(self.feed_forward(self.norm2(src)))
        # 在编码器输出前添加通道注意力
        # src = self.channel_attn(src)

        return src


class DecodingLayer(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dim_feedforward: int,
                 dropout: float, use_mask: bool, attention_type: str, with_bias: bool):
        super().__init__()
        # 自注意力和交叉注意力层
        self.self_attn, self.cross_attn = AttentionRegistry.get_attention(attention_type, dim_attention, num_heads, dropout, use_mask, with_bias)

        # 模块化前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_attention, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_attention)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(dim_attention)
        self.norm2 = nn.LayerNorm(dim_attention)
        self.norm3 = nn.LayerNorm(dim_attention)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # 通道注意力
        # self.channel_attn = ChannelAttention1D(dim_attention) # 仅多变量时需要

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播, pre-norm architecture
        :param tgt: target tensor from input or previous decodinglayer
        :param memory: key-value pair tensor from encoder
        :return: tgt: decoded tensor
        """
        # Pre-norm architecture
        tgt = tgt + self.dropout1(self.self_attn(self.norm1(tgt)))
        tgt = tgt + self.dropout2(self.cross_attn(self.norm2(tgt), memory, mask))
        tgt = tgt + self.dropout3(self.feed_forward(self.norm3(tgt)))
        # tgt = self.channel_attn(tgt)

        return tgt


class Encoder(nn.Module):
    def __init__(self, num_layers: int, dim_attention: int, num_heads: int, dim_feedforward: int,
                 dropout: float, attention_types: Union[str, List[str], List[List[str]]], with_bias: bool):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_layers = num_layers
        # 确定注意力配置类型， 如果传入的是一维数组，那么是串联，如果是高维维数组或者包含高维数组，那么是并联
        self.is_single = isinstance(attention_types, str)
        self.is_serial = isinstance(attention_types, list) and not any(isinstance(i, list) for i in attention_types)
        self.is_parallel = isinstance(attention_types, list) and any(isinstance(i, list) for i in attention_types)

        self.layers, self.norm, self.fusion = self.set_up_attention_layers(num_layers, dim_attention, num_heads, dim_feedforward, dropout, attention_types, with_bias)

    def set_up_attention_layers(self, num_layers: int, dim_attention: int, num_heads: int, dim_feedforward: int,
                 dropout: float, attention_types: Union[str, List[str], List[List[str]]], with_bias) -> (nn.ModuleList, nn.Module, nn.Module): # type: ignore
        if self.is_single:
            # 仅有一种注意力机制，所有层都使用相同的注意力机制，数量为num_layers
            return nn.ModuleList([
                EncodingLayer(dim_attention, num_heads, dim_feedforward, dropout, attention_types, with_bias) for _ in range(num_layers)
            ]), nn.LayerNorm(dim_attention), None
        elif self.is_serial:
            # 有多种注意力机制，但是每层仅有一种注意力机制，串联，数量为num_layers
            # if len(attention_types) != num_layers:
            #     raise ValueError("Attention type list must have the same length as the number of layers")
            return nn.ModuleList([
                EncodingLayer(dim_attention, num_heads, dim_feedforward, dropout, att_type, with_bias) for att_type in attention_types
            ]), nn.LayerNorm(dim_attention), None
        elif self.is_parallel:
            # 将多个串联进行并联，串联层的数量为num_layers，并联的数量为len(attention_types)
            return nn.ModuleList([
                nn.ModuleList([
                EncodingLayer(dim_attention, num_heads, dim_feedforward, dropout, serial_att_type, with_bias) for serial_att_type in parallel_att_types
                ]) for parallel_att_types in attention_types
            ]), nn.LayerNorm(dim_attention), FeatureFusion1D(dim_attention*len(attention_types), dim_attention)


    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param src: source tensor
        :return: src: encoded tensor
        """
        if self.is_single or self.is_serial:
            for layer in self.layers:
                src = layer(src)
            src = self.norm(src)
            return src
        elif self.is_parallel:
            sub_outputs = []
            for para_layers in self.layers:
                src_serial = src.clone()
                for serial_sub_layer in para_layers:
                    src_serial = serial_sub_layer(src_serial)
                sub_outputs.append(src_serial)
            src = self.fusion(sub_outputs)
            src = self.norm(src)
            return src
        else:
            raise ValueError("Unsupported attention type configuration. Please check the attention_types parameter.")


class Decoder(nn.Module):
    def __init__(self, num_layers: int, dim_attention: int, num_heads: int, dim_feedforward: int,
                 dropout: float, use_mask: bool, attention_types: Union[str, List[str], List[List[str]]], with_bias: bool):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_layers = num_layers
        # 确定注意力配置类型， 如果传入的是一维数组，那么是串联，如果是高维维数组或者包含高维数组，那么是并联
        self.is_single = isinstance(attention_types, str)
        self.is_serial = isinstance(attention_types, list) and not any(isinstance(i, list) for i in attention_types)
        self.is_parallel = isinstance(attention_types, list) and any(isinstance(i, list) for i in attention_types)

        self.layers, self.norm, self.fusion = self.set_up_attention_layers(num_layers, dim_attention, num_heads, dim_feedforward, dropout,
                                                   use_mask, attention_types, with_bias)

    def set_up_attention_layers(self, num_layers: int, dim_attention: int, num_heads: int, dim_feedforward: int,
                 dropout: float, use_mask: bool, attention_types: Union[str, List[str], List[List[str]]], with_bias) -> (nn.ModuleList, nn.Module, nn.Module): # type: ignore
        if self.is_single:
            # 仅有一种注意力机制，所有层都使用相同的注意力机制，数量为num_layers
            return nn.ModuleList([
                DecodingLayer(dim_attention, num_heads, dim_feedforward, dropout, use_mask, attention_types, with_bias) for _ in range(num_layers)
            ]), nn.LayerNorm(dim_attention), None
        elif self.is_serial:
            # 有多种注意力机制，但是每层仅有一种注意力机制，串联，数量为num_layers
            # if len(attention_types) != num_layers:
            #     raise ValueError("Attention type list must have the same length as the number of layers")
            return nn.ModuleList([
                DecodingLayer(dim_attention, num_heads, dim_feedforward, dropout, use_mask, att_type, with_bias) for att_type in attention_types
            ]), nn.LayerNorm(dim_attention), None
        elif self.is_parallel:
            # 将多个串联进行并联，串联层的数量为num_layers，并联的数量为len(attention_types)
            return nn.ModuleList([
                nn.ModuleList([
                DecodingLayer(dim_attention, num_heads, dim_feedforward, dropout, use_mask, serial_att_type, with_bias) for serial_att_type in parallel_att_types
                ]) for parallel_att_types in attention_types
            ]), nn.LayerNorm(dim_attention), FeatureFusion1D(dim_attention*len(attention_types), dim_attention)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param src: source tensor
        :return: src: encoded tensor
        """
        if self.is_single or self.is_serial:
            for layer in self.layers:
                tgt = layer(tgt, memory, mask)
            tgt = self.norm(tgt)
            return tgt
        elif self.is_parallel:
            sub_outputs = []
            tgt_serial = tgt.clone()
            for para_layers in self.layers:
                for serial_sub_layer in para_layers:
                    tgt_serial = serial_sub_layer(tgt_serial, memory, mask)
                sub_outputs.append(tgt_serial)
            tgt = self.fusion(sub_outputs)
            tgt = self.norm(tgt)
            return tgt
        else:
            raise ValueError("Unsupported attention type configuration. Please check the attention_types parameter.")

