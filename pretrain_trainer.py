import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
import os
import time
import logging
from tqdm import tqdm
from typing import Dict, Tuple, Optional
from gpt_model import GPTModel
from byte_level_bpe import ByteLevelBPETokenizer
from data_loader import create_pretrain_dataloader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pretrain.log"), logging.StreamHandler()]
)


class GPTTrainer:
    def __init__(
            self,
            model: GPTModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            device: torch.device,
            output_dir: str = "models/pretrain",
            lr: float = 2e-4,
            weight_decay: float = 0.1,
            betas: Tuple[float, float] = (0.9, 0.95),
            max_grad_norm: float = 1.0,
            num_epochs: int = 10,
            warmup_steps: int = 1000,
            log_interval: int = 100,
            save_interval: int = 1000
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 优化器（AdamW，带权重衰减）
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )

        # 学习率调度器（余弦退火+预热）
        total_steps = num_epochs * len(train_loader)
        self.scheduler = self._create_scheduler(total_steps, warmup_steps)

        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval

        # 混合精度训练
        self.scaler = GradScaler()

        # 记录最佳验证损失
        self.best_val_loss = float("inf")

    def _create_scheduler(self, total_steps: int, warmup_steps: int) -> LambdaLR:
        """创建学习率调度器：先线性预热，再余弦衰减"""

        def lr_lambda(step: int) -> float:
            # 预热阶段
            if step < warmup_steps:
                return step / warmup_steps
            # 余弦衰减阶段
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

        return LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练"""
        self.model.train()
        self.optimizer.zero_grad()

        # 数据移至设备
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 混合精度前向传播
        with autocast():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs["loss"]

        # 反向传播+梯度裁剪
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        # 参数更新
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

    def _val_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步验证"""
        self.model.eval()
        with torch.no_grad():
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            return outputs["loss"].item()

    def save_model(self, step: int, is_best: bool = False) -> None:
        """保存模型权重"""
        save_path = os.path.join(self.output_dir, f"pretrain_step_{step}.pth")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "step": step
        }, save_path)
        logging.info(f"模型保存至：{save_path}")

        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.output_dir, "pretrain_best.pth")
            torch.save({"model_state_dict": self.model.state_dict()}, best_path)
            logging.info(f"最佳模型保存至：{best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """加载 checkpoint 续训"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"]
        logging.info(f"从 checkpoint 加载成功：{checkpoint_path}，起始步数：{start_step}")
        return start_step

    def train(self, resume_from: Optional[str] = None) -> None:
        """主训练循环"""
        start_epoch = 0
        start_step = 0

        # 加载 checkpoint
        if resume_from is not None and os.path.exists(resume_from):
            start_step = self.load_checkpoint(resume_from)
            start_epoch = start_step // len(self.train_loader)

        total_steps = self.num_epochs * len(self.train_loader)
        global_step = start_step

        logging.info(f"开始训练：总 epoch={self.num_epochs}，总步数={total_steps}，起始步数={start_step}")

        for epoch in range(start_epoch, self.num_epochs):
            epoch_start_time = time.time()
            train_losses = []

            # 训练阶段
            train_pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")
            for batch in train_pbar:
                loss = self._train_step(batch)
                train_losses.append(loss)
                global_step += 1

                # 日志输出
                if global_step % self.log_interval == 0:
                    avg_loss = sum(train_losses[-self.log_interval:]) / self.log_interval
                    lr = self.optimizer.param_groups[0]["lr"]
                    logging.info(
                        f"Step {global_step}/{total_steps} | "
                        f"Train Loss: {avg_loss:.4f} | "
                        f"LR: {lr:.6f}"
                    )
                    train_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                # 保存 checkpoint
                if global_step % self.save_interval == 0:
                    self.save_model(global_step, is_best=False)

            # 验证阶段
            val_losses = []
            val_pbar = tqdm(self.val_loader, desc="Validation")
            for batch in val_pbar:
                val_loss = self._val_step(batch)
                val_losses.append(val_loss)
                val_pbar.set_postfix({"val_loss": f"{val_loss:.4f}"})

            avg_val_loss = sum(val_losses) / len(val_losses)
            epoch_time = (time.time() - epoch_start_time) / 60  # 分钟
            logging.info(
                f"Epoch {epoch + 1} 完成 | "
                f"Avg Train Loss: {sum(train_losses) / len(train_losses):.4f} | "
                f"Avg Val Loss: {avg_val_loss:.4f} | "
                f"耗时: {epoch_time:.2f}分钟"
            )

            # 保存最佳模型
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_model(global_step, is_best=True)

        logging.info("训练完成！")


# 预训练启动入口
def main():
    # 配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 50000
    d_model = 768
    n_layers = 12
    n_heads = 12
    d_ff = 3072
    max_seq_len = 2048
    batch_size = 8
    num_epochs = 10
    train_data_path = "data/pretrain/train.txt"
    val_data_path = "data/pretrain/val.txt"
    tokenizer_dir = "models/bpe_tokenizer"
    output_dir = "models/pretrain"

    # 加载分词器
    tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_dir)

    # 创建数据加载器
    train_loader = create_pretrain_dataloader(
        file_path=train_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len
    )
    val_loader = create_pretrain_dataloader(
        file_path=val_data_path,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_seq_len=max_seq_len
    )

    # 初始化模型
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )

    # 初始化训练器并启动训练
    trainer = GPTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        num_epochs=num_epochs
    )
    trainer.train()


if __name__ == "__main__":
    main()
