import torch
import torch.nn as nn
from gpt_model import GPTModel
from typing import Dict, Any


def quantize_model_int8(model: GPTModel) -> GPTModel:
    """将模型量化为INT8（仅权重，激活保持FP16）"""
    quantized_model = model.to(dtype=torch.float16)  # 激活用FP16

    # 遍历所有线性层，量化权重
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.Linear):
            # 权重从FP16量化为INT8（缩放至[-127, 127]）
            weight = module.weight.data
            scale = torch.max(torch.abs(weight)) / 127.0  # 缩放因子
            weight_int8 = torch.round(weight / scale).to(torch.int8)

            # 替换为量化权重，并保存缩放因子（推理时需用）
            module.weight = nn.Parameter(weight_int8, requires_grad=False)
            module.register_buffer("scale", scale)  # 保存缩放因子

    return quantized_model


def int8_linear_forward(self, x: torch.Tensor) -> torch.Tensor:
    """替换Linear层的forward方法，支持INT8权重推理"""
    # INT8权重 → 转换为FP16并乘以缩放因子
    weight = self.weight.to(torch.float16) * self.scale
    return torch.nn.functional.linear(x, weight, self.bias)


def enable_int8_inference(model: GPTModel) -> GPTModel:
    """启用INT8推理（替换Linear层的forward方法）"""
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "scale"):
            module.forward = int8_linear_forward.__get__(module, nn.Linear)
    return model


def save_quantized_model(model: GPTModel, save_path: str) -> None:
    """保存量化模型（包含权重和缩放因子）"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "quantization": "int8"
    }, save_path)


def load_quantized_model(
        model_config: Dict[str, Any],
        model_path: str,
        device: torch.device
) -> GPTModel:
    """加载量化模型并启用INT8推理"""
    model = GPTModel(**model_config).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = enable_int8_inference(model)
    return model


# 量化示例
def main():
    from byte_level_bpe import ByteLevelBPETokenizer

    # 加载原始模型
    tokenizer = ByteLevelBPETokenizer.from_pretrained("models/bpe_tokenizer")
    model = GPTModel(
        vocab_size=len(tokenizer.vocab),
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072
    )
    model.load_state_dict(torch.load("models/sft/sft_best.pth", map_location="cpu")["model_state_dict"])

    # 量化为INT8
    quantized_model = quantize_model_int8(model)
    quantized_model = enable_int8_inference(quantized_model)

    # 保存量化模型（体积约为原始的1/4）
    save_quantized_model(quantized_model, "models/sft/sft_best_int8.pth")
    print("INT8量化模型保存完成")


if __name__ == "__main__":
    main()
