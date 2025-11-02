# GPT模型复现与部署框架

本项目是一个基于Transformer解码器架构的GPT模型全流程实现，涵盖从**数据预处理、模型训练（预训练+微调）、性能评估到工程化部署**的完整链路。支持Byte-level BPE分词、自回归文本生成、模型量化优化及容器化服务部署，可作为基础框架快速适配不同场景的文本生成任务。


## 核心功能

- **Byte-level BPE分词器**：支持自定义词表大小，适配多语言文本编码。
- **Decoder-only Transformer**：实现标准GPT架构（多头注意力+前馈网络+残差连接）。
- **全流程训练**：
  - 预训练：基于因果语言模型（CLM）目标，支持大规模文本续训。
  - 微调：包含监督微调（SFT）和人类反馈强化学习（RLHF）。
- **推理优化**：
  - INT8量化：减少75%显存占用，保持生成质量。
  - KV缓存：推理速度提升2-3倍，适合实时交互场景。
- **工程化部署**：提供FastAPI接口和Docker容器配置，支持GPU加速。


## 环境要求

| 类别       | 具体要求                          |
|------------|-----------------------------------|
| 硬件       | NVIDIA GPU（≥12GB显存，支持CUDA）|
| 系统       | Ubuntu 20.04/22.04 或 WSL2       |
| 软件依赖   | Python 3.9、CUDA 11.7、PyTorch 1.13+ |
| 可选工具   | Docker、Docker Compose（容器化）  |


## 快速开始

### 1. 克隆仓库并安装依赖

```bash
# 克隆代码
git clone https://github.com/hongyuxu0/gpt.git
cd gpt

# 安装依赖（国内源加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
