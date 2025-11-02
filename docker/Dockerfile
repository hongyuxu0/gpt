# 基础镜像：NVIDIA CUDA 11.7 + Ubuntu 20.04
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 设置Python默认版本
RUN ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# 安装Python依赖（国内源加速）
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建工作目录
WORKDIR /app

# 复制项目文件
COPY . /app/

# 暴露API端口
EXPOSE 8000

# 启动命令：使用UVicorn运行FastAPI服务
CMD ["uvicorn", "inference.api_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
