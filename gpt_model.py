import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_components import TransformerDecoderLayer
from typing import Optional
from typing import List, Dict, Tuple


class GPTModel(nn.Module):
    def __init__(
            self,
            vocab_size: int = 50000,
            d_model: int = 768,
            n_layers: int = 12,  # GPT-2 small=12
            n_heads: int = 12,
            d_ff: int = 3072,
            max_seq_len: int = 2048,
            dropout: float = 0.1,
            pad_token_id: int = 1
    ):
        """GPT模型整体定义（Decoder-only架构）"""
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        # 1. 输入嵌入层（词嵌入+位置嵌入）
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)  # 可学习位置嵌入
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Transformer解码器堆叠
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 3. 输出层（预测下一个token）
        self.norm_final = nn.LayerNorm(d_model, eps=1e-6)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)
        # 输出层权重与词嵌入层共享（GPT优化策略）
        self.output_layer.weight = self.token_embedding.weight

        # 初始化参数
        self._init_weights()

    def _init_weights(self) -> None:
        """参数初始化（Xavier均匀初始化）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.d_model ** -0.5)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None  # 用于计算CLM损失
    ) -> Dict[str, torch.Tensor]:
        """
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: (batch_size, seq_len)
        :param labels: 标签（与input_ids同形，用于自回归损失计算）
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. 生成位置索引（0~seq_len-1）
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

        # 2. 输入嵌入（词嵌入+位置嵌入）
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        pos_emb = self.position_embedding(position_ids)  # (batch_size, seq_len, d_model)
        x = self.embedding_dropout(token_emb + pos_emb)

        # 3. 处理attention_mask（适配多头注意力输入格式）
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)  # (batch_size, 1, seq_len)

        # 4. 解码器堆叠前向传播
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, attention_mask)

        # 5. 输出层计算logits
        x = self.norm_final(x)
        logits = self.output_layer(x)  # (batch_size, seq_len, vocab_size)

        # 6. 计算CLM损失（预训练目标）
        loss = None
        if labels is not None:
            # 自回归损失：用第i个token预测第i+1个token，故偏移一位
            shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
            shift_labels = labels[:, 1:].contiguous()  # (batch_size, seq_len-1)

            # 忽略pad_token的损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
            loss = loss_fct(shift_logits.view(-1, shift_logits.shape[-1]), shift_labels.view(-1))

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_gen_len: int = 100,
            top_k: int = 50,  # Top-K采样
            temperature: float = 1.0  # 温度系数
    ) -> List[int]:
        """文本生成（自回归解码）"""
        self.eval()
        device = input_ids.device
        seq_len = input_ids.shape[1]

        # 生成循环（最多生成max_gen_len个token）
        for _ in range(max_gen_len):
            # 限制输入长度（避免超过位置嵌入范围）
            if seq_len >= self.position_embedding.num_embeddings:
                break

            # 前向传播获取logits
            outputs = self(input_ids=input_ids)
            logits = outputs["logits"][:, -1, :]  # 取最后一个token的logits (1, vocab_size)

            # Top-K采样（过滤低概率token）
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                logits = torch.full_like(logits, -1e9, device=device)
                logits.scatter_(-1, top_k_indices, top_k_values)

            # 温度缩放（降低随机性）
            if temperature != 1.0:
                logits = logits / temperature

            # 计算概率并采样
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (1,)

            # 终止条件（生成</s>则停止）
            if next_token_id.item() == self.token_embedding.vocab["</s>"]:
                break

            # 拼接新token
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            seq_len += 1

        return input_ids.squeeze(0).tolist()  # 转为列表返回
