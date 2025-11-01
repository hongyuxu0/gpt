import json
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


class ByteLevelBPETokenizer:
    def __init__(self, vocab_size: int = 50000, special_tokens: Dict[str, int] = None):
        """初始化Byte-level BPE分词器"""
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "<s>": 0,
            "<pad>": 1,
            "</s>": 2,
            "<unk>": 3
        }
        self.vocab: Dict[str, int] = self.special_tokens.copy()
        self.merges: Dict[Tuple[str, str], str] = {}  # (a,b) -> ab
        self.reverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.byte_encoder = self._build_byte_encoder()

    def _build_byte_encoder(self) -> Dict[int, str]:
        """构建字节到字符串的映射（Byte-level编码基础）"""
        byte_encoder = {}
        for i in range(256):
            if i < 32 or i > 126:
                byte_encoder[i] = f"<0x{i:02X}>"
            else:
                byte_encoder[i] = chr(i)
        return byte_encoder

    def _bytes_to_tokens(self, text: str) -> List[str]:
        """将文本转换为字节级token列表"""
        bytes_list = text.encode("utf-8")
        return [self.byte_encoder[b] for b in bytes_list]

    def train(self, texts: List[str], min_frequency: int = 2) -> None:
        """训练BPE分词器"""
        # 1. 初始化基础词表（字节级）
        token_counts = defaultdict(int)
        for text in texts:
            tokens = self._bytes_to_tokens(text)
            for token in tokens:
                token_counts[token] += 1

        # 添加字节级token到词表
        next_id = len(self.vocab)
        for token in token_counts:
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.reverse_vocab[next_id] = token
                next_id += 1
                if next_id >= self.vocab_size:
                    break

        # 2. 迭代合并高频子词对
        current_tokens = {text: self._bytes_to_tokens(text) for text in texts}
        while next_id < self.vocab_size:
            # 统计子词对频率
            pair_counts = defaultdict(int)
            for tokens in current_tokens.values():
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] += 1

            # 过滤低频对子
            valid_pairs = {p: c for p, c in pair_counts.items() if c >= min_frequency}
            if not valid_pairs:
                break  # 无更多有效合并

            # 选择频率最高的对子
            best_pair = max(valid_pairs, key=valid_pairs.get)
            merged_token = "".join(best_pair)

            # 更新词表和合并规则
            self.merges[best_pair] = merged_token
            self.vocab[merged_token] = next_id
            self.reverse_vocab[next_id] = merged_token
            next_id += 1

            # 更新所有文本的token序列
            for text in current_tokens:
                tokens = current_tokens[text]
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                current_tokens[text] = new_tokens

        print(f"BPE训练完成：词表大小={len(self.vocab)}，合并规则数={len(self.merges)}")

    def encode(self, text: str, max_seq_len: int = 2048, add_special_tokens: bool = True) -> Dict[str, List[int]]:
        """文本编码（返回input_ids和attention_mask）"""
        # 1. 字节级转换与合并
        tokens = self._bytes_to_tokens(text)
        i = 0
        while i < len(tokens) - 1:
            pair = (tokens[i], tokens[i + 1])
            if pair in self.merges:
                tokens = tokens[:i] + [self.merges[pair]] + tokens[i + 2:]
            else:
                i += 1

        # 2. 添加特殊token
        if add_special_tokens:
            tokens = ["<s>"] + tokens + ["</s>"]

        # 3. 转换为ID并处理长度
        input_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        if len(input_ids) > max_seq_len:
            input_ids = input_ids[:max_seq_len]  # 截断
        elif len(input_ids) < max_seq_len:
            input_ids += [self.vocab["<pad>"]] * (max_seq_len - len(input_ids))  # 填充

        # 4. 生成attention_mask（0=pad）
        attention_mask = [1 if id != self.vocab["<pad>"] else 0 for id in input_ids]

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, input_ids: List[int]) -> str:
        """ID序列解码为文本"""
        tokens = []
        for id in input_ids:
            token = self.reverse_vocab.get(id, "<unk>")
            if token in ["<s>", "</s>", "<pad>"]:
                continue
            tokens.append(token)

        # 字节级token还原为原始字节
        byte_str = ""
        for token in tokens:
            if token.startswith("<0x") and token.endswith(">"):
                byte_str += chr(int(token[3:-1], 16))
            else:
                byte_str += token

        return byte_str

    def save(self, save_dir: str) -> None:
        """保存词表和合并规则"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        with open(f"{save_dir}/vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with open(f"{save_dir}/merges.json", "w", encoding="utf-8") as f:
            json.dump({f"{k[0]},{k[1]}": v for k, v in self.merges.items()}, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(cls, save_dir: str) -> "ByteLevelBPETokenizer":
        """加载预训练分词器"""
        with open(f"{save_dir}/vocab.json", "r", encoding="utf-8") as f:
            vocab = json.load(f)
        with open(f"{save_dir}/merges.json", "r", encoding="utf-8") as f:
            merges_dict = json.load(f)
        merges = {tuple(k.split(",")): v for k, v in merges_dict.items()}

        tokenizer = cls()
        tokenizer.vocab = vocab
        tokenizer.merges = merges
        tokenizer.reverse_vocab = {v: k for k, v in vocab.items()}
        return tokenizer
