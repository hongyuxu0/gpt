from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import os
from typing import Dict, Optional
from gpt_model import GPTModel
from byte_level_bpe import ByteLevelBPETokenizer
from kv_cache_infer import generate_with_kv_cache
from quantizer import load_quantized_model

# 初始化FastAPI
app = FastAPI(title="GPT Inference Service")

# 全局变量：模型和分词器
model = None
tokenizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 请求体模型
class GenerateRequest(BaseModel):
    prefix: str  # 输入前缀（提示词）
    max_gen_len: int = 100  # 最大生成长度
    top_k: int = 50  # Top-K采样
    temperature: float = 0.8  # 温度系数


# 加载模型和分词器
def load_model_and_tokenizer():
    global model, tokenizer

    # 加载分词器
    tokenizer_dir = "models/bpe_tokenizer"
    tokenizer = ByteLevelBPETokenizer.from_pretrained(tokenizer_dir)
    print(f"分词器加载完成：{tokenizer_dir}")

    # 模型配置
    model_config = {
        "vocab_size": len(tokenizer.vocab),
        "d_model": 768,
        "n_layers": 12,
        "n_heads": 12,
        "d_ff": 3072,
        "max_seq_len": 2048
    }

    # 优先加载量化模型（如果存在）
    model_path = "models/sft/sft_best_int8.pth"
    if not os.path.exists(model_path):
        model_path = "models/sft/sft_best.pth"  # 否则加载原始模型

    if "int8" in model_path:
        # 加载INT8量化模型
        model = load_quantized_model(
            model_config=model_config,
            model_path=model_path,
            device=device
        )
    else:
        # 加载原始模型
        model = GPTModel(**model_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

    model.eval()
    print(f"模型加载完成：{model_path}，设备：{device}")


# 启动时加载模型
@app.on_event("startup")
def startup_event():
    load_model_and_tokenizer()


# 健康检查接口
@app.get("/health")
def health_check() -> Dict[str, str]:
    if model is not None and tokenizer is not None:
        return {"status": "healthy", "model_status": "loaded", "device": str(device)}
    else:
        return {"status": "unhealthy", "model_status": "not_loaded"}


# 生成接口
@app.post("/generate")
def generate_text(request: GenerateRequest) -> Dict[str, str]:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="模型未加载")

    try:
        # 调用带KV缓存的生成函数
        generated_text = generate_with_kv_cache(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prefix,
            max_gen_len=request.max_gen_len,
            top_k=request.top_k,
            temperature=request.temperature,
            device=device
        )
        return {"prefix": request.prefix, "generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败：{str(e)}")


# 主函数（本地测试用）
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
