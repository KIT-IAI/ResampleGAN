"""
GAN模型的生成器，支持动态配置的注意力机制（串联/并联）
"""
import torch.nn as nn
from typing import Dict, Type
from ResampleGAN.TransGAN.Attention.OriginalAttention import MultiHeadAttention, MultiHeadCrossAttention
from ResampleGAN.TransGAN.Attention.ConvAttention import MultiHeadTrendAwareAttention, MultiHeadTrendAwareCrossAttention
from ResampleGAN.TransGAN.Attention.FreqAttention import FreqAttention, FreqCrossAttention

class AttentionRegistryClass:
    """注意力机制注册表，用于动态注册和获取不同类型的注意力机制"""
    _attention_types: Dict[str, Dict[str, Type[nn.Module]]] = {
        'self': {},      # 自注意力机制
        'cross': {}      # 交叉注意力机制
    }

    @classmethod
    def register(cls, attention_type: str, self_attention: Type[nn.Module], cross_attention: Type[nn.Module]):
        """
        注册新的注意力机制
        :param attention_type: 注意力机制名称
        :param self_attention: 自注意力机制类
        :param cross_attention: 交叉注意力机制类
        """
        cls._attention_types['self'][attention_type] = self_attention
        cls._attention_types['cross'][attention_type] = cross_attention

    @classmethod
    def get_attention(cls, attention_type: str, dim_attention: int,
                      num_heads: int, dropout: float, with_bias: bool, use_mask: bool = None) -> (nn.Module, nn.Module):
        """
        获取指定的注意力机制
        :param attention_type: 注意力机制名称
        :return: 注意力机制类
        """
        if attention_type not in cls._attention_types['self']:
            raise ValueError(f"Unknown attention type: {attention_type}")

        return cls._attention_types['self'][attention_type](dim_attention = dim_attention, num_heads = num_heads, dropout = dropout, with_bias=with_bias), \
               cls._attention_types['cross'][attention_type](dim_attention = dim_attention, num_heads = num_heads, dropout = dropout, use_mask = use_mask, with_bias=with_bias)

# 注册不同类型的注意力机制
AttentionRegistry = AttentionRegistryClass()
AttentionRegistry.register('original', MultiHeadAttention, MultiHeadCrossAttention)
AttentionRegistry.register('conv', MultiHeadTrendAwareAttention, MultiHeadTrendAwareCrossAttention)
AttentionRegistry.register('freq', FreqAttention, FreqCrossAttention)