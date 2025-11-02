import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1):
        """多头自注意力（Decoder-only核心）"""
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model}需被n_heads={n_heads}整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 单头维度

        # 线性投影层（Q/K/V共享权重）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  # 缩放因子

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            causal_mask: bool = True  # 因果掩码（防止未来信息泄露）
    ) -> torch.Tensor:
        """
        :param x: 输入张量 (batch_size, seq_len, d_model)
        :param attention_mask: padding掩码 (batch_size, 1, seq_len)
        """
        batch_size = x.shape[0]

        # 1. 线性投影 + 多头拆分 (batch_size, n_heads, seq_len, d_k)
        q = self.w_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算注意力得分（缩放点积）
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch_size, n_heads, seq_len, seq_len)

        # 3. 应用因果掩码（下三角可见）
        if causal_mask:
            seq_len = x.shape[1]
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
            attn_scores = attn_scores.masked_fill(~mask, -1e9)  # 不可见位置设为-∞

        # 4. 应用padding掩码
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)

        # 5. Softmax + Dropout + 加权求和
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, d_k)

        # 6. 多头合并 + 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, n_heads, d_k)
        attn_output = attn_output.view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        return self.w_o(attn_output)


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int = 768, d_ff: int = 3072, dropout: float = 0.1):
        """前馈网络（d_ff=4*d_model）"""
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # GPT-2及后续版本使用GELU激活
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 768, n_heads: int = 12, d_ff: int = 3072, dropout: float = 0.1):
        """单个Transformer解码器层（残差连接+层归一化）"""
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)  # 预归一化（GPT-2采用）
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """预归一化流程：LayerNorm → Attention/FFN → 残差连接"""
        # 1. 自注意力子层
        x_norm1 = self.norm1(x)
        attn_output = self.attn(x_norm1, attention_mask, causal_mask=True)
        x = x + self.dropout(attn_output)

        # 2. 前馈子层
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        x = x + self.dropout(ffn_output)

        return x
