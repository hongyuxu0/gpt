import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import os
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from gpt_model import GPTModel
from byte_level_bpe import ByteLevelBPETokenizer
from data_loader import create_sft_dataloader, GPTPPODataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("finetune.log"), logging.StreamHandler()]
)


def sft_finetune(
        pretrained_model_path: str,
        sft_data_path: str,
        tokenizer: ByteLevelBPETokenizer,
        output_dir: str = "models/sft",
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        batch_size: int = 4,
        lr: float = 3e-5,
        num_epochs: int = 3,
        device: Optional[torch.device] = None
) -> None:
    """SFT（监督微调）实现"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型并加载预训练权重
    model = GPTModel(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    ).to(device)
    pretrained_state = torch.load(pretrained_model_path, map_location=device)["model_state_dict"]
    model.load_state_dict(pretrained_state)
    logging.info(f"加载预训练模型：{pretrained_model_path}")

    # 2. 创建数据加载器
    train_loader = create_sft_dataloader(
        file_path=sft_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len
    )

    # 3. 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader))
    scaler = GradScaler()
    best_loss = float("inf")

    # 4. 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        pbar = tqdm(train_loader, desc=f"SFT Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # 已处理：指令部分为-100

            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logging.info(f"SFT Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(output_dir, "sft_best.pth")
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            logging.info(f"最佳SFT模型保存至：{save_path}")

    logging.info("SFT微调完成！")


class RewardModel(nn.Module):
    """RLHF奖励模型（对生成文本打分）"""

    def __init__(self, base_model: GPTModel, hidden_size: int = 256):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.d_model, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)  # 输出标量奖励分
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            prompt_len: int  # 指令部分长度（仅对响应部分打分）
    ) -> torch.Tensor:
        """
        :return: 奖励分 (batch_size,)
        """
        batch_size = input_ids.shape[0]
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs["logits"]  # (batch_size, seq_len, vocab_size) → 复用logits的隐藏状态

        # 仅取响应部分的最后一个token的隐藏状态作为特征
        response_last_idx = attention_mask.sum(dim=1) - 1  # 每个样本响应的最后位置
        response_features = last_hidden_state[torch.arange(batch_size), response_last_idx]  # (batch_size, d_model)

        rewards = self.reward_head(response_features).squeeze(-1)  # (batch_size,)
        return rewards


def train_reward_model(
        sft_model_path: str,
        pairwise_data_path: str,
        tokenizer: ByteLevelBPETokenizer,
        output_dir: str = "models/reward",
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        batch_size: int = 4,
        lr: float = 1e-5,
        num_epochs: int = 3,
        device: Optional[torch.device] = None
) -> None:
    """训练奖励模型（RLHF第一步）"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载SFT模型作为基础
    base_model = GPTModel(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    ).to(device)
    sft_state = torch.load(sft_model_path, map_location=device)["model_state_dict"]
    base_model.load_state_dict(sft_state)

    # 2. 初始化奖励模型
    reward_model = RewardModel(base_model).to(device)

    # 3. 加载pairwise数据
    dataset = GPTPPODataset(
        file_path=pairwise_data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 4. 训练配置
    optimizer = optim.AdamW(reward_model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()  # 二分类损失（判断chosen是否优于rejected）
    best_loss = float("inf")

    # 5. 训练循环
    for epoch in range(num_epochs):
        reward_model.train()
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Reward Model Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            chosen_mask = (chosen_ids != tokenizer.vocab["<pad>"]).long()
            rejected_mask = (rejected_ids != tokenizer.vocab["<pad>"]).long()
            prompt_len = batch["prompt_len"].to(device)

            # 计算奖励分
            chosen_rewards = reward_model(chosen_ids, chosen_mask, prompt_len)
            rejected_rewards = reward_model(rejected_ids, rejected_mask, prompt_len)

            # 损失：希望chosen奖励 > rejected奖励（标签为1）
            logits = chosen_rewards - rejected_rewards  # (batch_size,)
            labels = torch.ones_like(logits, device=device)  # 标签：chosen应更优
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logging.info(f"Reward Model Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(output_dir, "reward_best.pth")
            torch.save({"model_state_dict": reward_model.state_dict()}, save_path)
            logging.info(f"最佳奖励模型保存至：{save_path}")

    logging.info("奖励模型训练完成！")


def rlhf_ppo(
        sft_model_path: str,
        reward_model_path: str,
        pairwise_data_path: str,
        tokenizer: ByteLevelBPETokenizer,
        output_dir: str = "models/rlhf",
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 2048,
        batch_size: int = 2,
        ppo_epochs: int = 5,
        clip_epsilon: float = 0.2,  # PPO剪辑系数
        gamma: float = 0.99,  # 奖励折扣因子
        lam: float = 0.95,  # GAE系数
        device: Optional[torch.device] = None
) -> None:
    """PPO优化（RLHF第二步）"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # 1. 初始化策略模型（待优化）和参考模型（固定）
    policy_model = GPTModel(
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    ).to(device)
    ref_model = GPTModel(  # 参考模型（固定参数，用于计算优势）
        vocab_size=len(tokenizer.vocab),
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    ).to(device)

    # 加载SFT权重初始化
    sft_state = torch.load(sft_model_path, map_location=device)["model_state_dict"]
    policy_model.load_state_dict(sft_state)
    ref_model.load_state_dict(sft_state)
    ref_model.eval()  # 参考模型不更新

    # 2. 加载奖励模型
    reward_model = RewardModel(ref_model).to(device)  # 复用参考模型的基础结构
    reward_state = torch.load(reward_model_path, map_location=device)["model_state_dict"]
    reward_model.load_state_dict(reward_state)
    reward_model.eval()

    # 3. 数据加载
    dataset = GPTPPODataset(
        file_path=pairwise_data_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # 4. 优化器
    optimizer = optim.AdamW(policy_model.parameters(), lr=3e-6, weight_decay=0.01)

    # 5. PPO循环
    for ppo_epoch in range(ppo_epochs):
        logging.info(f"PPO Epoch {ppo_epoch + 1}/{ppo_epochs}")
        total_ppo_loss = 0.0

        for batch in tqdm(dataloader, desc="PPO Steps"):
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_mask = batch["prompt_mask"].to(device)
            prompt_len = batch["prompt_len"].to(device)
            batch_size = prompt_ids.shape[0]

            # a. 策略模型生成响应
            policy_model.eval()
            with torch.no_grad():
                # 从prompt开始生成响应（最大生成50token）
                generated_ids = policy_model.generate(
                    input_ids=prompt_ids,
                    max_gen_len=50,
                    top_k=50,
                    temperature=0.7
                )
                # 拼接prompt和生成的响应
                full_ids = torch.tensor(generated_ids, device=device).unsqueeze(0)  # (1, full_seq_len)
                full_mask = (full_ids != tokenizer.vocab["<pad>"]).long()

            # b. 计算策略分布和参考分布
            policy_model.train()
            policy_logits = policy_model(input_ids=full_ids)["logits"]  # (1, seq_len, vocab_size)
            with torch.no_grad():
                ref_logits = ref_model(input_ids=full_ids)["logits"]  # 参考模型分布

            # 提取响应部分的logits（仅响应部分参与策略更新）
            response_logits = policy_logits[:, prompt_len[0]:-1, :]  # (1, response_len-1, vocab_size)
            response_ids = full_ids[:, prompt_len[0] + 1:]  # (1, response_len-1)
            ref_response_logits = ref_logits[:, prompt_len[0]:-1, :]

            # 计算策略概率和参考概率
            policy_probs = torch.gather(F.softmax(response_logits, dim=-1), -1, response_ids.unsqueeze(-1)).squeeze(-1)
            ref_probs = torch.gather(F.softmax(ref_response_logits, dim=-1), -1, response_ids.unsqueeze(-1)).squeeze(-1)
            ratio = policy_probs / (ref_probs + 1e-8)  # 重要性权重

            # c. 计算奖励和优势
            with torch.no_grad():
                rewards = reward_model(full_ids, full_mask, prompt_len)  # (1,)
                # 简化：使用单步奖励作为优势（实际应使用GAE）
                advantages = rewards.repeat(response_ids.shape[1])  # (response_len-1,)

            # d. PPO损失（剪辑版）
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
            ppo_loss = -torch.min(surr1, surr2).mean()  # 负号：最大化奖励

            # e. 优化策略模型
            optimizer.zero_grad()
            ppo_loss.backward()
            optimizer.step()

            total_ppo_loss += ppo_loss.item()

        avg_ppo_loss = total_ppo_loss / len(dataloader)
        logging.info(f"PPO Epoch {ppo_epoch + 1} | Avg PPO Loss: {avg_ppo_loss:.4f}")

        # 保存模型
        save_path = os.path.join(output_dir, f"ppo_epoch_{ppo_epoch + 1}.pth")
        torch.save({"model_state_dict": policy_model.state_dict()}, save_path)
        logging.info(f"PPO模型保存至：{save_path}")

    # 保存最终最佳模型
    final_path = os.path.join(output_dir, "ppo_best.pth")
    torch.save({"model_state_dict": policy_model.state_dict()}, final_path)
    logging.info(f"最终PPO模型保存至：{final_path}")
    logging.info("RLHF-PPO优化完成！")


# 微调入口（支持SFT和RLHF）
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["sft", "rlhf"])
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, help="预训练模型路径（SFT用）")
    parser.add_argument("--sft_model", type=str, help="SFT模型路径（RLHF用）")
    parser.add_argument("--output_dir", type=str, default="models/finetune")
    args = parser.parse_args()

    # 加载分词器
    tokenizer = ByteLevelBPETokenizer.from_pretrained("models/bpe_tokenizer")

    if args.mode == "sft":
        sft_finetune(
            pretrained_model_path=args.pretrained_model,
            sft_data_path=args.data_path,
            tokenizer=tokenizer,
            output_dir=args.output_dir
        )
    elif args.mode == "rlhf":
        # 先训练奖励模型，再进行PPO
        reward_dir = os.path.join(args.output_dir, "reward")
        train_reward_model(
            sft_model_path=args.sft_model,
            pairwise_data_path=args.data_path,
            tokenizer=tokenizer,
            output_dir=reward_dir
        )
        rlhf_ppo(
            sft_model_path=args.sft_model,
            reward_model_path=os.path.join(reward_dir, "reward_best.pth"),
            pairwise_data_path=args.data_path,
            tokenizer=tokenizer,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    main()
