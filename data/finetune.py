#!/usr/bin/env python3
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from datasets import Dataset as HFDataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("finetune")


class ModelArgs:
    def __init__(self, ns):
        self.model_name_or_path = ns.model
        self.trust_remote_code = ns.trust_remote_code
        self.use_lora = ns.use_lora
        self.lora_r = ns.lora_r
        self.lora_alpha = ns.lora_alpha
        self.lora_dropout = ns.lora_dropout


class DataArgs:
    def __init__(self, ns):
        self.dataset_path = ns.dataset
        self.max_seq_length = ns.context_size
        self.preprocessing_num_workers = ns.preprocessing_workers


class TrainArgs(TrainingArguments):
    pass


class MemoryCallback(TrainerCallback):
    def __init__(self, every=10):
        self.every = every
        self._t0 = None

    def on_step_begin(self, args, state, control, **kwargs):
        if self._t0 is None:
            self._t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self._t0 and state.global_step % self.every == 0:
            if torch.cuda.is_available() and state.is_world_process_zero:
                alloc = torch.cuda.memory_allocated() / 2 ** 30
                reserv = torch.cuda.memory_reserved() / 2 ** 30
                logger.info(
                    f"step={state.global_step} time={time.time() - self._t0:.2f}s gpu_alloc={alloc:.2f}GB gpu_resv={reserv:.2f}GB"
                )
            self._t0 = None


class TextGenCallback(TrainerCallback):
    def __init__(self, tokenizer: PreTrainedTokenizer, prompts: List[str], every: int, **gen_kw):
        self.tok = tokenizer
        self.prompts = prompts
        self.every = every
        self.gen_kw = gen_kw

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Only generate text and log on the main process
        if not state.is_world_process_zero:
            return

        if state.global_step % self.every or model is None:
            return
        model.eval()
        with torch.no_grad():
            for i, p in enumerate(self.prompts):
                inputs = self.tok(p, return_tensors="pt").to(model.device)
                out = model.generate(**inputs, **self.gen_kw)
                txt = self.tok.decode(out[0], skip_special_tokens=True)
                logger.info(f"\nPrompt {i + 1}: {p}\nGenerated: {txt}\n")
                if wandb.run:
                    wandb.log({f"gen_{i}": txt}, step=state.global_step)
        model.train()


def load_dataset_local(data_args: DataArgs, tok: PreTrainedTokenizer, cache: Optional[str]) -> HFDataset:
    p = Path(data_args.dataset_path)
    if p.is_dir():
        files = list(p.glob("**/*.txt"))
        logger.info(f"Found {len(files)} text files in {p}")
        texts = [f.read_text("utf-8") for f in files]
        ds = HFDataset.from_dict({"text": texts})
    else:
        ds = load_dataset("text", data_files=str(p), cache_dir=cache)["train"]

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, padding="max_length", max_length=data_args.max_seq_length)

    # When using smaller datasets, it's better to disable multiprocessing for tokenization
    num_proc = data_args.preprocessing_num_workers if Path(data_args.dataset_path).is_dir() else 1
    return ds.map(tok_fn, batched=True, num_proc=num_proc, remove_columns=ds.column_names)


def build_model(model_args: ModelArgs, cache: Optional[str]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=cache,
                                        trust_remote_code=model_args.trust_remote_code, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # FIX: device_map must be None when using DeepSpeed/FSDP.
    # The distributed training framework is responsible for device placement.
    dev_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.float16,
        device_map=dev_map,
    )

    if model_args.use_lora:
        cfg = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    return model, tok


def main():
    rank_env = int(os.environ.get("LOCAL_RANK", "-1"))
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    # data
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--context-size", type=int, default=2048)
    parser.add_argument("--preprocessing-workers", type=int, default=1)
    # train
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-path", default="./outputs")
    parser.add_argument("--cache", default="./cache")
    parser.add_argument("--logs", default="./logs")
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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--zero-stage", type=int, default=3)
    parser.add_argument("--local-rank", type=int, default=rank_env)
    # generation / wandb
    parser.add_argument("--prompt-every", type=int, default=1000)
    parser.add_argument("--prompt-tokens", type=int, default=200)
    parser.add_argument("--project-id", default="mistral-finetune")

    ns = parser.parse_args()

    # FIX: Set the WANDB_PROJECT environment variable.
    # The Trainer will automatically pick this up for logging.
    os.environ["WANDB_PROJECT"] = ns.project_id

    if ns.local_rank != -1:
        torch.cuda.set_device(ns.local_rank)
        dist.init_process_group("nccl")

    random.seed(ns.seed)
    np.random.seed(ns.seed)
    torch.manual_seed(ns.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ns.seed)

    # FIX: Removed manual wandb.init(). The Trainer will handle it.
    global_rank = int(os.environ.get("RANK", "0"))

    model, tok = build_model(ModelArgs(ns), cache=ns.cache)
    data = load_dataset_local(DataArgs(ns), tok, cache=ns.cache)

    if ns.bs == -1:
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            bs_guess = 4 if mem > 80 else 2 if mem > 40 else 1
            ns.bs = max(1, int(bs_guess / ns.bs_divisor))
            logger.info(f"Auto-detected batch size: {ns.bs}")
        else:
            ns.bs = 1
            logger.info("CUDA not available. Setting batch size to 1.")

    out_dir = Path(ns.output_path) / f"results-{ns.run_name}"

    targs = TrainArgs(
        output_dir=str(out_dir),
        overwrite_output_dir=not ns.no_resume,
        num_train_epochs=ns.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=ns.gradients,
        warmup_ratio=ns.warmup_ratio,
        learning_rate=ns.lr,
        fp16=ns.fp16,
        logging_dir=ns.logs,
        logging_steps=ns.logging_steps,
        save_steps=ns.save_steps,
        eval_steps=ns.eval_steps,
        save_total_limit=3,
        report_to=["wandb"],
        run_name=ns.run_name,
        seed=ns.seed,
        local_rank=ns.local_rank,
        ddp_find_unused_parameters=False,
        deepspeed=None if ns.zero_stage == 0 else {
            "zero_optimization": {
                "stage": ns.zero_stage,
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                "offload_param": {"device": "cpu", "pin_memory": True} if ns.zero_stage == 3 else None,
                "overlap_comm": True,
                "contiguous_gradients": True,
            },
            "gradient_accumulation_steps": ns.gradients,
            "gradient_clipping": 1.0,
        },
    )

    collate = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False, pad_to_multiple_of=8)

    cbs = [MemoryCallback(ns.logging_steps)]
    if ns.prompt_every > 0:
        prompts = [
            "The key to successful machine learning is",
            "In the future, artificial intelligence will",
            "The most important consideration when training models is",
        ]
        cbs.append(TextGenCallback(tok, prompts, ns.prompt_every, max_new_tokens=ns.prompt_tokens))

    trainer = Trainer(model=model, args=targs, train_dataset=data, data_collator=collate, tokenizer=tok, callbacks=cbs)
    logger.info("Training started")
    trainer.train(resume_from_checkpoint=False)

    # The Trainer will handle saving the model correctly on the main process.
    # We just need to save the tokenizer and the ready file.
    if trainer.is_world_process_zero():
        final = out_dir / "final"
        trainer.save_model(str(final))
        tok.save_pretrained(str(final))
        (final / ".ready.txt").write_text("done\n")


if __name__ == "__main__":
    sys.exit(main())