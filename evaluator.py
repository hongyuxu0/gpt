import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from typing import Optional
from typing import List, Dict, Tuple
from gpt_model import GPTModel
from byte_level_bpe import ByteLevelBPETokenizer
from data_loader import GPTPretrainDataset, GPTSFTDataset


class GPTEvaluator:
    def __init__(self, model: GPTModel, tokenizer: ByteLevelBPETokenizer, device: Optional[torch.device] = None):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.rouge = Rouge()
        self.smoother = SmoothingFunction().method4  # BLEU平滑函数

    @torch.no_grad()
    def compute_perplexity(self, dataloader: DataLoader) -> float:
        """计算困惑度（PPL，越低越好）"""
        total_loss = 0.0
        total_tokens = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]

            # 统计有效token数（非pad）
            token_count = attention_mask.sum().item() - input_ids.shape[0]  # 减去<s>的计数
            total_loss += loss.item() * token_count
            total_tokens += token_count

        ppl = np.exp(total_loss / total_tokens)
        return ppl

    @torch.no_grad()
    def generate_predictions(
            self,
            prompts: List[str],
            max_gen_len: int = 100,
            top_k: int = 50,
            temperature: float = 0.8
    ) -> List[str]:
        """生成预测文本"""
        predictions = []
        for prompt in prompts:
            # 编码prompt
            encoding = self.tokenizer.encode(text=prompt, max_seq_len=len(prompt) + 10, add_special_tokens=True)
            input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long).unsqueeze(0).to(self.device)

            # 生成
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_gen_len=max_gen_len,
                top_k=top_k,
                temperature=temperature
            )

            # 解码（去除prompt部分）
            full_text = self.tokenizer.decode(generated_ids)
            pred_text = full_text[len(prompt):].strip()  # 只保留生成的响应部分
            predictions.append(pred_text)

        return predictions

    def compute_bleu(self, predictions: List[str], references: List[List[str]]) -> float:
        """计算BLEU分数（0~1，越高越好）"""
        # 分词（按空格，实际应与模型分词一致）
        pred_tokens = [pred.lower().split() for pred in predictions]
        ref_tokens = [[ref.lower().split() for ref in refs] for refs in references]

        # 计算BLEU-4
        bleu_scores = []
        for pred, refs in zip(pred_tokens, ref_tokens):
            score = sentence_bleu(refs, pred, smoothing_function=self.smoother)
            bleu_scores.append(score)

        return np.mean(bleu_scores)

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """计算ROUGE分数（R-1/R-2/R-L，0~1，越高越好）"""
        rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

        for pred, ref in zip(predictions, references):
            try:
                scores = self.rouge.get_scores(pred.lower(), ref.lower())[0]
                rouge_scores["rouge-1"].append(scores["rouge-1"]["f"])
                rouge_scores["rouge-2"].append(scores["rouge-2"]["f"])
                rouge_scores["rouge-l"].append(scores["rouge-l"]["f"])
            except:
                # 处理空字符串等异常
                for key in rouge_scores:
                    rouge_scores[key].append(0.0)

        # 取平均值
        return {k: np.mean(v) for k, v in rouge_scores.items()}

    def evaluate_pretrained(self, test_data_path: str, max_seq_len: int = 2048) -> Dict[str, float]:
        """评估预训练模型（仅PPL）"""
        dataset = GPTPretrainDataset(
            file_path=test_data_path,
            tokenizer=self.tokenizer,
            max_seq_len=max_seq_len
        )
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
        ppl = self.compute_perplexity(dataloader)
        return {"perplexity": ppl}

    def evaluate_sft(
            self,
            test_data_path: str,
            max_seq_len: int = 2048,
            max_gen_len: int = 100
    ) -> Dict[str, float]:
        """评估SFT模型（PPL+BLEU+ROUGE）"""
        # 1. 加载测试数据
        with open(test_data_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        prompts = [item["instruction"] for item in test_data]
        references = [item["response"] for item in test_data]
        ref_for_bleu = [[ref] for ref in references]  # BLEU需要列表的列表

        # 2. 生成预测
        predictions = self.generate_predictions(prompts, max_gen_len=max_gen_len)

        # 3. 计算指标
        bleu = self.compute_bleu(predictions, ref_for_bleu)
        rouge = self.compute_rouge(predictions, references)

        # 4. 计算PPL（可选，需加载SFT测试集）
        sft_dataset = GPTSFTDataset(
            file_path=test_data_path,
            tokenizer=self.tokenizer,
            max_seq_len=max_seq_len
        )
        sft_dataloader = DataLoader(sft_dataset, batch_size=8, shuffle=False, num_workers=4)
        ppl = self.compute_perplexity(sft_dataloader)

        return {
            "perplexity": ppl,
            "bleu": bleu,
            "rouge-1": rouge["rouge-1"],
            "rouge-2": rouge["rouge-2"],
            "rouge-l": rouge["rouge-l"]
        }


# 评估入口
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["pretrain", "sft"], default="sft")
    args = parser.parse_args()

    # 加载分词器
    tokenizer = ByteLevelBPETokenizer.from_pretrained("models/bpe_tokenizer")

    # 加载模型
    model = GPTModel(
        vocab_size=len(tokenizer.vocab),
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu")["model_state_dict"])

    # 评估
    evaluator = GPTEvaluator(model, tokenizer)
    if args.model_type == "pretrain":
        metrics = evaluator.evaluate_pretrained(args.test_data_path)
        print(f"预训练模型评估：PPL={metrics['perplexity']:.2f}")
    else:
        metrics = evaluator.evaluate_sft(args.test_data_path)
        print(f"SFT模型评估：")
        print(f"  PPL: {metrics['perplexity']:.2f}")
        print(f"  BLEU: {metrics['bleu']:.4f}")
        print(f"  ROUGE-1: {metrics['rouge-1']:.4f}")
        print(f"  ROUGE-2: {metrics['rouge-2']:.4f}")
        print(f"  ROUGE-L: {metrics['rouge-l']:.4f}")


if __name__ == "__main__":
    main()
