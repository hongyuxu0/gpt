# GPT模型复现实现

本项目基于Transformer解码器架构，完整复现了GPT模型的训练与部署流程，包括：
- Byte-level BPE分词器
- Decoder-only Transformer架构
- 预训练（CLM目标）
- 微调（SFT+RLHF）
- 性能评估（PPL、BLEU、ROUGE）
- 推理优化（量化、KV缓存）与部署（Docker+FastAPI）


## 环境要求
- 硬件：NVIDIA GPU（推荐≥12GB显存，如RTX 3090/4090）
- 系统：Ubuntu 20.04/22.04（或WSL2）
- 软件：Python 3.9、CUDA 11.7、Docker（可选）


## 安装依赖
```bash
# 克隆仓库
git clone https://github.com/your-username/gpt-reproduction.git
cd gpt-reproduction

# 安装依赖（国内源加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
