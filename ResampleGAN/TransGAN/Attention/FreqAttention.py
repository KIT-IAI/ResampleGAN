"""
频域注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FreqAttention(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dropout: float, with_bias: bool):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_heads = num_heads
        self.head_dim = dim_attention // num_heads
        assert dim_attention % num_heads == 0, "dim_attention must be divisible by num_heads"

        # 分别定义处理实部和虚部的Q,K,V投影层
        self.W_q_r = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_q_i = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_k_r = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_k_i = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_v_r = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_v_i = nn.Linear(dim_attention, dim_attention, bias=with_bias)

        self.W_o = nn.Linear(dim_attention, dim_attention, bias=False)

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
        B, L, E = x.shape

        # 应用Layer Normalization
        x = self.norm(x)

        # 对时序维度进行FFT变换到频域：x_freq: [B, L', E] with complex values
        # L' = L//2 + 1
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')  # [B, L', E]
        #
        # # 将E拆分为H,D：x_freq [B, L', H, D]
        # x_freq = rearrange(x_freq, 'b l (h d) -> b l h d', h=self.num_heads)

        # 分别对实部和虚部投影
        Q_r = self.W_q_r(x_freq.real)  # [B, L', H, D]
        Q_i = self.W_q_i(x_freq.imag)
        Q_freq = torch.complex(Q_r, Q_i)

        K_r = self.W_k_r(x_freq.real)
        K_i = self.W_k_i(x_freq.imag)
        K_freq = torch.complex(K_r, K_i)

        V_r = self.W_v_r(x_freq.real)
        V_i = self.W_v_i(x_freq.imag)
        V_freq = torch.complex(V_r, V_i)

        Q_freq = rearrange(Q_freq, 'b l (h d) -> b l h d', h=self.num_heads)
        K_freq = rearrange(K_freq, 'b l (h d) -> b l h d', h=self.num_heads)
        V_freq = rearrange(V_freq, 'b l (h d) -> b l h d', h=self.num_heads)

        # 计算注意力分数
        # 实部: Qr, Qi, Kr, Ki
        Qr, Qi = Q_freq.real, Q_freq.imag
        Kr, Ki = K_freq.real, K_freq.imag

        # attn_scores: [B,H,L',L']
        attn_scores = torch.einsum('b l h d, b m h d -> b h l m', Qr, Kr) + \
                      torch.einsum('b l h d, b m h d -> b h l m', Qi, Ki)

        # 缩放注意力分数并应用softmax
        scaling_factor = self.head_dim ** -0.5
        attn_weights = F.softmax(attn_scores * scaling_factor, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和得到输出
        Vr, Vi = V_freq.real, V_freq.imag
        out_freq_real = torch.einsum('b h l m, b m h d -> b h l d', attn_weights, Vr)
        out_freq_imag = torch.einsum('b h l m, b m h d -> b h l d', attn_weights, Vi)

        out_freq = torch.complex(out_freq_real, out_freq_imag)  # [B,H,L',D]

        # 回到 [B,L',E]
        out_freq = rearrange(out_freq, 'b h l d -> b l (h d)')

        # iFFT恢复时域
        out_time = torch.fft.irfft(out_freq, n=L, dim=1, norm='ortho')  # [B,L,E]

        # 输出投影
        out = self.W_o(out_time)
        out = self.proj_dropout(out)

        return out


class FreqCrossAttention(nn.Module):
    def __init__(self, dim_attention: int, num_heads: int, dropout: float, use_mask: bool, with_bias: bool):
        super().__init__()
        self.dim_attention = dim_attention
        self.num_heads = num_heads
        self.head_dim = dim_attention // num_heads
        assert dim_attention % num_heads == 0, "dim_attention must be divisible by num_heads"
        self.use_mask = use_mask

        # 分别定义处理实部和虚部的Q,K,V投影层
        self.W_q_r = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_q_i = nn.Linear(dim_attention, dim_attention, bias=with_bias)

        self.W_k_r = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_k_i = nn.Linear(dim_attention, dim_attention, bias=with_bias)

        self.W_v_r = nn.Linear(dim_attention, dim_attention, bias=with_bias)
        self.W_v_i = nn.Linear(dim_attention, dim_attention, bias=with_bias)

        self.W_o = nn.Linear(dim_attention, dim_attention, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.norm = nn.LayerNorm(dim_attention)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        :param query: [B, Lq, E]
        :param key_value: [B, Lkv, E]
        :param mask: [B, 1, Lq, Lkv], 若use_mask为True则为因果掩码，否则为None
        :return: [B, Lq, E]
        """
        B, Lq, E = query.shape
        Lkv = key_value.shape[1]

        # 对query进行LayerNorm
        query = self.norm(query)

        # 对 query 和 key_value 分别进行FFT变换到频域
        # query_freq: [B, Lq', E], key_freq: [B, Lkv', E], value_freq: [B, Lkv', E]
        query_freq = torch.fft.rfft(query, dim=1, norm='ortho')
        kv_freq = torch.fft.rfft(key_value, dim=1, norm='ortho')

        # # 重排维度：将E分解成H,D
        # # query_freq: [B, Lq', H, D], kv_freq: [B, Lkv', H, D]
        # query_freq = rearrange(query_freq, 'b l (h d) -> b l h d', h=self.num_heads)
        # kv_freq = rearrange(kv_freq, 'b l (h d) -> b l h d', h=self.num_heads)

        # 分别对query_freq的实部、虚部进行线性投影，得到Q_freq
        Q_r = self.W_q_r(query_freq.real)
        Q_i = self.W_q_i(query_freq.imag)
        Q_freq = torch.complex(Q_r, Q_i)  # [B,Lq',H,D]

        # 分别对key_freq的实部、虚部进行线性投影，得到K_freq
        K_r = self.W_k_r(kv_freq.real)
        K_i = self.W_k_i(kv_freq.imag)
        K_freq = torch.complex(K_r, K_i)  # [B,Lkv',H,D]

        # 分别对value_freq的实部、虚部进行线性投影，得到V_freq
        V_r = self.W_v_r(kv_freq.real)
        V_i = self.W_v_i(kv_freq.imag)
        V_freq = torch.complex(V_r, V_i)  # [B,Lkv',H,D]

        Q_freq = rearrange(Q_freq, 'b l (h d) -> b l h d', h=self.num_heads)
        K_freq = rearrange(K_freq, 'b l (h d) -> b l h d', h=self.num_heads)
        V_freq = rearrange(V_freq, 'b l (h d) -> b l h d', h=self.num_heads)

        # 计算注意力分数: Q与K的点积(实部虚部分别相乘后相加)
        Qr, Qi = Q_freq.real, Q_freq.imag
        Kr, Ki = K_freq.real, K_freq.imag

        # attn_scores: [B,H,Lq',Lkv']
        attn_scores = (torch.einsum('b l h d, b m h d -> b h l m', Qr, Kr) +
                       torch.einsum('b l h d, b m h d -> b h l m', Qi, Ki))

        # 缩放并加mask
        scaling_factor = self.head_dim ** -0.5
        attn_scores = attn_scores * scaling_factor

        if self.use_mask and mask is not None:
            # mask: [B, 1, Lq, Lkv]，需要和Lq',Lkv'映射关系？
            # 简化处理：mask 通常在时域生效。如果你想在频域中使用mask，
            # 需要确保mask对应频域长度。否则建议mask在计算attn_weights之后再加入。
            # 这里假设mask与频域维度等价使用不变(这可能并不严格合理)。
            attn_scores = attn_scores + mask

        # softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和
        Vr, Vi = V_freq.real, V_freq.imag
        out_freq_real = torch.einsum('b h l m, b m h d -> b h l d', attn_weights, Vr)
        out_freq_imag = torch.einsum('b h l m, b m h d -> b h l d', attn_weights, Vi)
        out_freq = torch.complex(out_freq_real, out_freq_imag)  # [B,H,Lq',D]

        # 回到[B,Lq',E]
        out_freq = rearrange(out_freq, 'b h l d -> b l (h d)')

        # iFFT恢复时域 [B,Lq,E]
        out_time = torch.fft.irfft(out_freq, n=Lq, dim=1, norm='ortho')

        # 输出投影
        out = self.W_o(out_time)
        out = self.proj_dropout(out)

        return out