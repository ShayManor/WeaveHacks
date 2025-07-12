FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Fix NVIDIA repository issues and install dependencies in one layer
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    ca-certificates \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install Python packages in one layer with specific versions
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.2.0 \
    transformers==4.38.1 \
    accelerate==0.27.2 \
    datasets==2.18.0 \
    evaluate==0.4.1 \
    wandb==0.16.3 \
    peft==0.8.2 \
    bitsandbytes==0.42.0 \
    scipy==1.12.0 \
    sentencepiece==0.1.99 \
    protobuf==4.25.3 \
    deepspeed==0.13.1 \
    safetensors==0.4.2 \
    einops==0.7.0

# Create app directory and copy files
WORKDIR /app
COPY finetune.py /app/
RUN chmod +x /app/finetune.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV TOKENIZERS_PARALLELISM=false

CMD ["python", "/app/finetune.py"]