import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
from typing import List, Dict, Tuple
from byte_level_bpe import ByteLevelBPETokenizer


class GPTPretrainDataset(Dataset):
    """预训练数据集（文本续写任务，CLM目标）"""

    def __init__(
            self,
            file_path: str,
            tokenizer: ByteLevelBPETokenizer,
            max_seq_len: int = 2048,
            min_text_len: int = 100  # 过滤过短文本
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_text_len = min_text_len

        # 加载并过滤文本
        with open(file_path, "r", encoding="utf-8") as f:
            self.texts = [line.strip() for line in f if len(line.strip()) >= min_text_len]
        print(f"加载预训练数据：{len(self.texts)}条样本，文件路径：{file_path}")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]

        # 随机截断（避免文本过长导致编码后超过max_seq_len）
        if len(text) > self.max_seq_len * 2:  # 假设平均每个token对应2个字符
            start_idx = random.randint(0, len(text) - self.max_seq_len * 2)
            text = text[start_idx:start_idx + self.max_seq_len * 2]

        # 编码（包含特殊token：<s>开头，</s>结尾）
        encoding = self.tokenizer.encode(
            text=text,
            max_seq_len=self.max_seq_len,
            add_special_tokens=True
        )

        # 标签与输入一致（CLM任务：用前i个token预测i+1个）
        return {
            "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(encoding["input_ids"], dtype=torch.long)  # 标签与输入相同
        }


class GPTSFTDataset(Dataset):
    """SFT微调数据集（指令-响应对任务）"""

    def __init__(
            self,
            file_path: str,
            tokenizer: ByteLevelBPETokenizer,
            max_seq_len: int = 2048,
            prompt_template: str = "### 指令：{instruction}\n### 响应：{response}"
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template

        # 加载SFT数据（格式：[{"instruction": "...", "response": "..."}]）
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"加载SFT数据：{len(self.data)}条样本，文件路径：{file_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        instruction = item["instruction"].strip()
        response = item["response"].strip()

        # 格式化prompt（指令+响应）
        full_text = self.prompt_template.format(instruction=instruction, response=response)

        # 编码完整文本
        encoding = self.tokenizer.encode(
            text=full_text,
            max_seq_len=self.max_seq_len,
            add_special_tokens=True
        )
        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        # 构建标签：仅计算响应部分的损失（指令部分标签设为-100，被CrossEntropyLoss忽略）
        prompt_text = self.prompt_template.format(instruction=instruction, response="")
        prompt_encoding = self.tokenizer.encode(
            text=prompt_text,
            max_seq_len=self.max_seq_len,
            add_special_tokens=True
        )
        prompt_len = sum(prompt_encoding["attention_mask"])  # 指令部分长度

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # 指令部分不参与损失计算

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class GPTPPODataset(Dataset):
    """RLHF-PPO数据集（ pairwise偏好数据）"""

    def __init__(
            self,
            file_path: str,
            tokenizer: ByteLevelBPETokenizer,
            max_seq_len: int = 2048,
            prompt_template: str = "### 指令：{instruction}\n### 响应："
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.prompt_template = prompt_template

        # 加载pairwise数据（格式：{"instruction": "...", "chosen": "...", "rejected": "..."}）
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        print(f"加载PPO数据：{len(self.data)}条样本，文件路径：{file_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        instruction = item["instruction"].strip()
        chosen_response = item["chosen"].strip()
        rejected_response = item["rejected"].strip()

        # 编码prompt（仅指令部分，用于生成响应）
        prompt_text = self.prompt_template.format(instruction=instruction)
        prompt_encoding = self.tokenizer.encode(
            text=prompt_text,
            max_seq_len=self.max_seq_len,
            add_special_tokens=True
        )
        prompt_ids = torch.tensor(prompt_encoding["input_ids"], dtype=torch.long)
        prompt_mask = torch.tensor(prompt_encoding["attention_mask"], dtype=torch.long)
        prompt_len = sum(prompt_encoding["attention_mask"])

        # 编码被偏好的响应和被拒绝的响应
        chosen_full = prompt_text + chosen_response
        chosen_encoding = self.tokenizer.encode(
            text=chosen_full,
            max_seq_len=self.max_seq_len,
            add_special_tokens=True
        )
        chosen_ids = torch.tensor(chosen_encoding["input_ids"], dtype=torch.long)

        rejected_full = prompt_text + rejected_response
        rejected_encoding = self.tokenizer.encode(
            text=rejected_full,
            max_seq_len=self.max_seq_len,
            add_special_tokens=True
        )
        rejected_ids = torch.tensor(rejected_encoding["input_ids"], dtype=torch.long)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "prompt_len": prompt_len,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids
        }


def create_pretrain_dataloader(
        file_path: str,
        tokenizer: ByteLevelBPETokenizer,
        batch_size: int = 8,
        max_seq_len: int = 2048,
        num_workers: int = 4
) -> DataLoader:
    """创建预训练数据加载器"""
    dataset = GPTPretrainDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


def create_sft_dataloader(
        file_path: str,
        tokenizer: ByteLevelBPETokenizer,
        batch_size: int = 8,
        max_seq_len: int = 2048,
        num_workers: int = 4
) -> DataLoader:
    """创建SFT微调数据加载器"""
    dataset = GPTSFTDataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
