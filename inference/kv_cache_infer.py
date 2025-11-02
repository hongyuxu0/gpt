import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from gpt_model import GPTModel
from byte_level_bpe import ByteLevelBPETokenizer


class KVCache:
    """KV缓存管理器（存储多头注意力的键值对，加速推理）"""

    def __init__(self, n_layers: int, n_heads: int, d_k: int, max_seq_len: int, device: torch.device):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        # 初始化缓存（layers x batch x heads x seq_len x d_k）
        self.k_cache = [
            torch.zeros(1, n_heads, 0, d_k, device=device)  # 初始序列长度为0
            for _ in range(n_layers)
        ]
        self.v_cache = [
            torch.zeros(1, n_heads, 0, d_k, device=device)
            for _ in range(n_layers)
        ]

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新指定层的KV缓存"""
        # k/v形状：(batch=1, heads, new_seq_len, d_k)
        self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
        self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)

        # 截断缓存至最大长度（防止OOM）
        if self.k_cache[layer_idx].shape[2] > self.max_seq_len:
            self.k_cache[layer_idx] = self.k_cache[layer_idx][:, :, -self.max_seq_len:, :]
            self.v_cache[layer_idx] = self.v_cache[layer_idx][:, :, -self.max_seq_len:, :]

        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def reset(self) -> None:
        """重置缓存（新序列生成前调用）"""
        for i in range(self.n_layers):
            self.k_cache[i] = torch.zeros(1, self.n_heads, 0, self.d_k, device=self.device)
            self.v_cache[i] = torch.zeros(1, self.n_heads, 0, self.d_k, device=self.device)


def patched_attention_forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = True,
        kv_cache: Optional[KVCache] = None,
        layer_idx: Optional[int] = None
) -> torch.Tensor:
    """
    替换MultiHeadAttention的forward方法，支持KV缓存
    :param kv_cache: KVCache实例
    :param layer_idx: 当前层索引
    """
    batch_size = x.shape[0]

    # 线性投影 + 多头拆分 (batch, heads, seq_len, d_k)
    q = self.w_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    k = self.w_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    v = self.w_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

    # 使用KV缓存（仅推理时）
    if kv_cache is not None and layer_idx is not None:
        k, v = kv_cache.update(layer_idx, k, v)  # 从缓存中获取历史KV并更新

    # 计算注意力得分
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch, heads, seq_len, seq_len_total)

    # 应用掩码（仅对新生成的token生效）
    if causal_mask:
        seq_len = x.shape[1]  # 新输入的长度（1，因为推理时逐token生成）
        total_seq_len = k.shape[2]  # 历史+新序列长度
        mask = torch.tril(torch.ones(seq_len, total_seq_len, device=x.device)).bool()
        attn_scores = attn_scores.masked_fill(~mask, -1e9)

    if attention_mask is not None:
        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)

    # 注意力计算
    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq_len, d_k)

    # 多头合并 + 输出投影
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    return self.w_o(attn_output)


def enable_kv_cache_inference(model: GPTModel, kv_cache: KVCache) -> None:
    """为模型启用KV缓存推理（替换注意力层的forward方法）"""
    for layer_idx, decoder_layer in enumerate(model.decoder_layers):
        # 替换多头注意力的forward方法，并绑定kv_cache和layer_idx
        def make_forward(layer_idx):
            def forward(x, attention_mask=None, causal_mask=True):
                return patched_attention_forward(
                    decoder_layer.attn,
                    x,
                    attention_mask=attention_mask,
                    causal_mask=causal_mask,
                    kv_cache=kv_cache,
                    layer_idx=layer_idx
                )

            return forward

        decoder_layer.attn.forward = make_forward(layer_idx)


@torch.no_grad()
def generate_with_kv_cache(
        model: GPTModel,
        tokenizer: ByteLevelBPETokenizer,
        prompt: str,
        max_gen_len: int = 100,
        top_k: int = 50,
        temperature: float = 0.8,
        device: torch.device = None
) -> str:
    """带KV缓存的文本生成（速度提升2-3倍）"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    # 初始化KV缓存
    n_layers = len(model.decoder_layers)
    n_heads = model.decoder_layers[0].attn.n_heads
    d_k = model.d_model // n_heads
    kv_cache = KVCache(
        n_layers=n_layers,
        n_heads=n_heads,
        d_k=d_k,
        max_seq_len=model.position_embedding.num_embeddings,
        device=device
    )
    enable_kv_cache_inference(model, kv_cache)

    # 编码prompt
    encoding = tokenizer.encode(text=prompt, add_special_tokens=True)
    input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long).unsqueeze(0).to(device)

    # 预热缓存（处理prompt）
    model(input_ids=input_ids, attention_mask=attention_mask)
    generated_ids = input_ids.squeeze(0).tolist()
    current_len = len(generated_ids)

    # 逐token生成
    for _ in range(max_gen_len):
        if current_len >= model.position_embedding.num_embeddings:
            break  # 超过最大长度

        # 取最后一个token作为输入
        last_token = torch.tensor([generated_ids[-1]], dtype=torch.long).unsqueeze(0).to(device)
        last_mask = torch.tensor([1], dtype=torch.long).unsqueeze(0).to(device)

        # 前向传播（仅处理新token，复用缓存）
        outputs = model(input_ids=last_token, attention_mask=last_mask)
        logits = outputs["logits"][:, -1, :]  # (1, vocab_size)

        # Top-K采样
        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
            logits = torch.full_like(logits, -1e9, device=device)
            logits.scatter_(-1, top_k_indices, top_k_values)

        # 温度缩放
        if temperature != 1.0:
            logits = logits / temperature

        # 采样下一个token
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        # 终止条件
        if next_token_id == tokenizer.vocab["</s>"]:
            break

        generated_ids.append(next_token_id)
        current_len += 1

    # 解码结果
    return tokenizer.decode(generated_ids)
