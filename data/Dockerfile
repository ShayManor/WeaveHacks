FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# --- system deps ---
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
 && apt-get update --fix-missing \
 && apt-get install -y --no-install-recommends git wget vim ca-certificates \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Python stack ---
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    torch==2.2.1+cu121 \
    transformers==4.41.1 \
    accelerate==0.28.0 \
    deepspeed==0.14.4 \
    peft==0.12.0 \
    datasets==2.18.0 evaluate==0.4.1 \
    wandb==0.16.3 scipy==1.12.0 sentencepiece==0.1.99 \
    protobuf==4.25.3 safetensors==0.4.2 einops==0.7.0

# --- app code ---
WORKDIR /app
COPY finetune.py .
RUN chmod +x finetune.py

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

CMD ["python", "finetune.py"]
