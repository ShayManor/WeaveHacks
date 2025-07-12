#!/usr/bin/env python3
import json
import gc
import os
import sys
import time
import math
import random
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
import wandb
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    PreTrainedTokenizer,
    PreTrainedModel,
)
from datasets import load_dataset, Dataset as HFDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to trust remote code"}
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use LoRA for efficient training"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    dataset_path: str = field(
        metadata={"help": "Path to the training dataset"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of processes to use for preprocessing"}
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    warmup_ratio: float = field(default=0.1)
    logging_steps: int = field(default=10)
    save_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)


class MemoryMonitorCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None and state.global_step % args.logging_steps == 0:
            elapsed = time.time() - self.start_time
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 ** 3
                reserved = torch.cuda.memory_reserved() / 1024 ** 3
                logger.info(
                    f"Step {state.global_step} - Time: {elapsed:.2f}s - "
                    f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
                )
            self.start_time = None


class TextGenerationCallback(TrainerCallback):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            prompts: List[str],
            generate_every: int = 1000,
            max_new_tokens: int = 200,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 50,
    ):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.generate_every = generate_every
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.generate_every == 0 and model is not None:
            model.eval()
            with torch.no_grad():
                for i, prompt in enumerate(self.prompts):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.info(f"\nPrompt {i + 1}: {prompt}\nGenerated: {generated_text}\n")

                    if wandb.run is not None:
                        wandb.log({
                            f"generation_{i}": wandb.Table(
                                columns=["step", "prompt", "generated"],
                                data=[[state.global_step, prompt, generated_text]]
                            )
                        }, step=state.global_step)
            model.train()


def load_and_prepare_dataset(
        data_args: DataArguments,
        tokenizer: PreTrainedTokenizer,
        cache_dir: Optional[str] = None,
) -> HFDataset:
    """Load and prepare dataset for training."""

    # Check if dataset path is a directory with text files
    dataset_path = Path(data_args.dataset_path)

    if dataset_path.is_dir():
        # Load all text files from directory
        text_files = list(dataset_path.glob("**/*.txt"))
        logger.info(f"Found {len(text_files)} text files in {dataset_path}")

        texts = []
        for file_path in text_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())

        # Create dataset from texts
        dataset = HFDataset.from_dict({"text": texts})
    else:
        # Try to load as a HuggingFace dataset
        dataset = load_dataset("text", data_files=str(dataset_path), cache_dir=cache_dir)
        dataset = dataset["train"]

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=data_args.max_seq_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset


def setup_model_and_tokenizer(
        model_args: ModelArguments,
        cache_dir: Optional[str] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Setup model and tokenizer with proper configurations."""

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
    )

    # Apply LoRA if requested
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    # Model arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--trust-remote-code", type=bool, default=False)
    parser.add_argument("--use-lora", type=bool, default=False)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Data arguments
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--preprocessing-workers", type=int, default=4)

    # Training arguments
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--output-path", type=str, default="./outputs")
    parser.add_argument("--cache", type=str, default="./cache")
    parser.add_argument("--logs", type=str, default="./logs")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--bs", type=int, default=-1)
    parser.add_argument("--bs-divisor", type=float, default=1.0)
    parser.add_argument("--gradients", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--no-resume", type=bool, default=False)
    parser.add_argument("--zero-stage", type=int, default=3)
    parser.add_argument("--local-rank", type=int, default=-1)

    # Generation arguments
    parser.add_argument("--prompt-every", type=int, default=1000)
    parser.add_argument("--prompt-tokens", type=int, default=200)
    parser.add_argument("--prompt-samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)

    # Parse arguments
    parser.add_argument("--eot", type=str, default="</s>")
    parser.add_argument("--pad", type=str, default="<unk>")
    parser.add_argument("--project-id", type=str, default="mistral-finetune")

    args = parser.parse_args()

    # Setup distributed training
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Initialize wandb
    if args.local_rank in [-1, 0]:
        wandb.init(
            project=args.project_id,
            name=args.run_name,
            config=vars(args),
        )

    # Create model and data arguments
    model_args = ModelArguments(
        model_name_or_path=args.model,
        trust_remote_code=args.trust_remote_code,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    data_args = DataArguments(
        dataset_path=args.dataset,
        max_seq_length=args.context_size,
        preprocessing_num_workers=args.preprocessing_workers,
    )

    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_args, cache_dir=args.cache)

    # Load and prepare dataset
    train_dataset = load_and_prepare_dataset(data_args, tokenizer, cache_dir=args.cache)

    # Auto batch size estimation
    if args.bs == -1:
        # Simple heuristic for batch size
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            if gpu_memory > 80:  # A100/H100
                args.bs = 4
            elif gpu_memory > 40:  # A40
                args.bs = 2
            else:
                args.bs = 1
        else:
            args.bs = 1

        args.bs = int(args.bs / args.bs_divisor)
        args.bs = max(1, args.bs)
        logger.info(f"Auto-detected batch size: {args.bs}")

    # Setup training arguments
    output_dir = os.path.join(args.output_path, f"results-{args.run_name}")

    training_args = CustomTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=not args.no_resume,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.gradients,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
        fp16=args.fp16,
        logging_dir=args.logs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to=["wandb"] if args.local_rank in [-1, 0] else [],
        run_name=args.run_name,
        seed=args.seed,
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        deepspeed={
            "zero_optimization": {
                "stage": args.zero_stage,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                } if args.zero_stage == 3 else None,
                "overlap_comm": True,
                "contiguous_gradients": True,
                "sub_group_size": 1e9,
                "reduce_bucket_size": "auto",
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_gather_16bit_weights_on_model_save": True,
            },
            "gradient_accumulation_steps": args.gradients,
            "gradient_clipping": 1.0,
            "steps_per_print": args.logging_steps,
            "train_batch_size": "auto",
            "train_micro_batch_size_per_gpu": "auto",
            "wall_clock_breakdown": False,
        } if args.zero_stage > 0 else None,
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    # Setup callbacks
    callbacks = [MemoryMonitorCallback()]

    # Add text generation callback
    sample_prompts = [
        "The key to successful machine learning is",
        "In the future, artificial intelligence will",
        "The most important consideration when training models is",
    ]

    if args.prompt_every > 0:
        callbacks.append(
            TextGenerationCallback(
                tokenizer=tokenizer,
                prompts=sample_prompts,
                generate_every=args.prompt_every,
                max_new_tokens=args.prompt_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Start training
    logger.info("Starting training...")
    trainer.train(resume_from_checkpoint=not args.no_resume)

    # Save final model
    if args.local_rank in [-1, 0]:
        final_path = os.path.join(output_dir, "final")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)

        # Mark as ready
        with open(os.path.join(final_path, ".ready.txt"), "w") as f:
            f.write("Model training completed successfully\n")

        logger.info(f"Model saved to {final_path}")

    # Cleanup
    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()