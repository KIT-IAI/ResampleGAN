"""
短期趋势注意力 https://github.com/guoshnBJTU/ASTGNN/blob/main/model/ASTGNN.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadTrendAwareAttention(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dropout: float, with_bias: bool,
                 kernel_size: int = 3):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_heads = num_heads
        self.head_dim = dim_attention // num_heads
        assert dim_attention % num_heads == 0, "dim_attention must be divisible by num_heads"

        # 时间趋势感知卷积层替代Q,K的线性投影
        padding = (kernel_size - 1) // 2
        self.query_conv = nn.Conv2d(dim_attention, dim_attention,
                                    kernel_size=(1, kernel_size),
                                    padding=(0, padding))
        self.key_conv = nn.Conv2d(dim_attention, dim_attention,
                                  kernel_size=(1, kernel_size),
                                  padding=(0, padding))

        # 只保留V的线性投影和输出投影
        self.W_v = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_o = nn.Linear(dim_attention, dim_attention, bias=with_bias)

        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer Normalization
        self.norm = nn.LayerNorm(dim_attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 应用Layer Normalization
        x = self.norm(x)

        # 重排张量用于卷积操作
        # [batch, seq_len, dim] -> [batch, dim, 1, seq_len]
        x_conv = rearrange(x, 'b n d -> b d 1 n')

        # 应用时间趋势感知卷积,直接得到Q,K
        # [batch, dim, 1, seq_len] -> [batch, seq_len, dim]
        Q = rearrange(self.query_conv(x_conv), 'b d 1 n -> b n d')
        K = rearrange(self.key_conv(x_conv), 'b d 1 n -> b n d')

        # 只对V使用线性投影
        V = self.W_v(x)

        # 重排维度为多头形式
        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.num_heads)

        # 计算注意力分数
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)

        # 缩放并应用softmax
        scaling_factor = self.head_dim ** -0.5
        attn_weights = F.softmax(attn_scores * scaling_factor, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 计算注意力输出
        out = torch.einsum('bhal,bhlv->bhav', attn_weights, V)

        # 重排维度并输出投影
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.W_o(out)
        out = self.proj_dropout(out)

        return out


class MultiHeadTrendAwareCrossAttention(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dropout: float, use_mask: bool, with_bias: bool, kernel_size: int = 3):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_heads = num_heads
        self.head_dim = dim_attention // num_heads
        assert dim_attention % num_heads == 0, "dim_attention must be divisible by num_heads"
        self.use_mask = use_mask

        # 时间趋势感知卷积层替代Q,K的线性投影
        padding = (kernel_size - 1) // 2
        self.query_conv = nn.Conv2d(dim_attention, dim_attention,
                                    kernel_size=(1, kernel_size),
                                    padding=(0, padding))
        self.key_conv = nn.Conv2d(dim_attention, dim_attention,
                                  kernel_size=(1, kernel_size),
                                  padding=(0, padding))

        # 只保留V的线性投影和输出投影
        self.W_v = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_o = nn.Linear(dim_attention, dim_attention, bias=with_bias)

        # Dropout层
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer Normalization
        self.norm = nn.LayerNorm(dim_attention)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        :param mask: cause mask matrix
        :param query: target tensor from decoder (B, Lq, E)
        :param key_value: memory tensor from encoder (B, Lkv, E)
        :return: output tensor
        """
        # 应用Layer Normalization到query
        query = self.norm(query)

        # 重排张量用于卷积操作
        # [batch, seq_len, dim] -> [batch, dim, 1, seq_len]
        q_conv = rearrange(query, 'b n d -> b d 1 n')
        kv_conv = rearrange(key_value, 'b n d -> b d 1 n')

        # 应用时间趋势感知卷积,直接得到Q,K
        # [batch, dim, 1, seq_len] -> [batch, seq_len, dim]
        Q = rearrange(self.query_conv(q_conv), 'b d 1 n -> b n d')
        K = rearrange(self.key_conv(kv_conv), 'b d 1 n -> b n d')

        # 对memory的value进行线性投影
        V = self.W_v(key_value)

        # 重排维度为多头形式
        Q = rearrange(Q, "b n (h d) -> b h n d", h=self.num_heads)
        K = rearrange(K, "b n (h d) -> b h n d", h=self.num_heads)
        V = rearrange(V, "b n (h d) -> b h n d", h=self.num_heads)

        # 计算注意力分数
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', Q, K)

        # 缩放
        scaling_factor = self.head_dim ** -0.5
        attn_scores = attn_scores * scaling_factor

        # 如果use_mask为True，应用因果掩码
        if self.use_mask and mask is not None:
            # mask shape: (B, 1, Lq, Lk)
            attn_scores = attn_scores + mask

        # 应用softmax和dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 计算注意力输出
        out = torch.einsum('bhal,bhlv->bhav', attn_weights, V)

        # 重排维度并输出投影
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.W_o(out)
        out = self.proj_dropout(out)

        return out