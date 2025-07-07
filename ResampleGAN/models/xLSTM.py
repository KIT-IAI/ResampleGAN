import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 模型定义（简洁 xLSTM）
class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super().__init__()
        assert in_features % num_blocks == 0 and out_features % num_blocks == 0
        self.blocks = nn.ModuleList([
            nn.Linear(in_features // num_blocks, out_features // num_blocks)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        split = torch.chunk(x, len(self.blocks), dim=-1)
        return torch.cat([blk(s) for blk, s in zip(self.blocks, split)], dim=-1)


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x + self.ffn(x)
        return x


class xLSTMBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, batch_first=True)
        self.attn = AttentionBlock(dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, state):
        out, new_state = self.lstm(x, state)
        out = self.norm(out)
        out = self.attn(out)
        return out, new_state


class xLSTM(nn.Module):
    def __init__(self, dim_input, dim_model, num_layers=2, num_heads=4):
        super().__init__()
        self.encoder = nn.Linear(dim_input, dim_model)
        self.blocks = nn.ModuleList([
            xLSTMBlock(dim_model, num_heads) for _ in range(num_layers)
        ])
        self.decoder = nn.Linear(dim_model, dim_input)

    def forward(self, x):
        x = self.encoder(x)
        state = None
        for block in self.blocks:
            x, state = block(x, state)
        x = self.decoder(x)
        return x
    

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out