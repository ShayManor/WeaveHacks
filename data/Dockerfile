FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers==4.36.2 \
    accelerate==0.25.0 \
    datasets==2.16.1 \
    evaluate==0.4.1 \
    wandb==0.16.2 \
    peft==0.7.1 \
    bitsandbytes==0.41.3 \
    scipy \
    sentencepiece \
    protobuf \
    deepspeed==0.12.6

# Create app directory
WORKDIR /app

# Copy the training script
COPY finetune.py /app/

# Make the script executable
RUN chmod +x /app/finetune.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

CMD ["python", "/app/finetune.py"]