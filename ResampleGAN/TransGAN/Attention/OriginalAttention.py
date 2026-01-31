"""
原版注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dropout: float, with_bias: bool):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_heads = num_heads
        self.head_dim = dim_attention // num_heads
        assert dim_attention % num_heads == 0, "dim_attention must be divisible by num_heads"

        # QKV投影矩阵
        self.W_q = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # Query投影
        self.W_k = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # Key投影
        self.W_v = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # Value投影
        self.W_o = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # 输出投影

        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer Normalization
        self.norm = nn.LayerNorm(dim_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        :param x: input tensor from input time series or previous encoding layer
        :return: out: output tensor
        """
        # 应用Layer Normalization
        x = self.norm(x)

        # 计算Q、K、V并重排维度
        # rearrange解释:
        # "b n (h d) -> b h n d" 意味着:
        # 1. 输入张量形状为 [batch, seq_len, (heads * head_dim)]
        # 2. 将最后一个维度拆分为 heads 和 head_dim
        # 3. 调整维度顺序为 [batch, heads, seq_len, head_dim]

        # Q, K, v变换: [B, L, E] -> [B, H, L, D]
        Q = rearrange(self.W_q(x), "b n (h d) -> b h n d", h=self.num_heads)  # b:batch, n:seq_len, h:heads, d:head_dim
        K = rearrange(self.W_k(x), "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(self.W_v(x), "b n (h d) -> b h n d", h=self.num_heads)

        # 计算注意力分数
        # einsum说明: 'bhqd,bhkd->bhqk'
        # b: batch维度对齐
        # h: head维度对齐
        # q: query序列长度
        # k: key序列长度
        # d: 进行点积的维度(head_dim)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)

        # 缩放注意力分数并应用softmax
        scaling_factor = self.head_dim ** -0.5
        attn_weights = F.softmax(attn_scores * scaling_factor, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 计算注意力输出
        # einsum说明: 'bhal,bhlv->bhav'
        # 将注意力权重与Value相乘并求和
        out = torch.einsum('bhal,bhlv->bhav', attn_weights, V)

        # 重排维度: [B, H, L, D] -> [B, L, E]
        # rearrange解释:
        # 1. 输入形状为 [batch, heads, seq_len, head_dim]
        # 2. 将heads和head_dim合并为embedding_dim
        # 3. 最终形状为 [batch, seq_len, embedding_dim]
        out = rearrange(out, "b h n d -> b n (h d)")

        # 输出投影和dropout
        out = self.W_o(out)
        out = self.proj_dropout(out)

        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dropout: float, use_mask: bool, with_bias: bool):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_heads = num_heads
        self.head_dim = dim_attention // num_heads
        assert dim_attention % num_heads == 0, "dim_attention must be divisible by num_heads"
        self.use_mask = use_mask

        # QKV投影矩阵
        self.W_q = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # Query投影
        self.W_k = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # Key投影
        self.W_v = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # Value投影
        self.W_o = nn.Linear(dim_attention, dim_attention, bias=with_bias)  # 输出投影

        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer Normalization
        self.norm = nn.LayerNorm(dim_attention)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        :param query: target tensor from initial output or previous decoding layer
        :param key_value: memory tensor from encoder
        :return:
        """
        # 应用Layer Normalization到query
        query = self.norm(query)

        # 计算Q、K、V并重排维度
        # rearrange解释:
        # 1. 将dim_attention维度拆分为 (num_heads, head_dim)
        # 2. 调整维度顺序为 [batch, heads, seq_len, head_dim]

        # Query变换: [B, Lq, E] -> [B, H, Lq, D]
        Q = rearrange(self.W_q(query), "b n (h d) -> b h n d", h=self.num_heads)  # b:batch, n:query_len, h:heads, d:head_dim
        # Key-value变换: [B, Lkv, E] -> [B, H, Lkv, D]
        K = rearrange(self.W_k(key_value), "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(self.W_v(key_value), "b n (h d) -> b h n d", h=self.num_heads)

        # 计算注意力分数
        # einsum说明: 'bhqd,bhkd->bhqk'
        # b: batch维度对齐
        # h: head维度对齐
        # q: query序列长度
        # k: key序列长度
        # d: 进行点积的维度(head_dim)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)
        scaling_factor = self.head_dim ** -0.5
        attn_scores = attn_scores * scaling_factor

        # 如果use_mask为True，应用因果掩码
        if self.use_mask and mask is not None:
            # mask shape: (B, 1, Lq, Lk)
            attn_scores = attn_scores + mask

        # 缩放注意力分数并应用softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 计算注意力输出
        # einsum说明: 'bhal,bhlv->bhav'
        # 将注意力权重与Value相乘并求和
        out = torch.einsum('bhal,bhlv->bhav', attn_weights, V)

        # 重排维度: [B, H, Lq, D] -> [B, Lq, E]
        # rearrange解释:
        # 1. 输入形状为 [batch, heads, query_len, head_dim]
        # 2. 将heads和head_dim合并为embedding_dim
        out = rearrange(out, "b h n d -> b n (h d)")

        # 输出投影和dropout
        out = self.W_o(out)
        out = self.proj_dropout(out)

        return out