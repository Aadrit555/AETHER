from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer


class SFTTrainer:
    def __init__(self, model_id: str, dataset: list[dict], output_dir: str | Path, learning_rate: float = 2e-4, epochs: int = 1, max_length: int = 512, gradient_accumulation: int = 8, device: str = "auto", resume_from: str | None = None, progress: Callable[[float, str], None] | None = None, max_steps: int = 5000, cancel_check: Callable[[], bool] | None = None):
        self.model_id = model_id
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_length = max_length
        self.gradient_accumulation = gradient_accumulation
        self.device = self._resolve_device(device)
        self.resume_from = resume_from
        self.progress = progress or (lambda *_: None)
        self.max_steps = max_steps
        self.cancel_check = cancel_check or (lambda: False)

    def train(self) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=os.getenv("HF_TOKEN") or None)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id, dtype=dtype, token=os.getenv("HF_TOKEN") or None)
        model.to(self.device)

        target_modules = self._find_lora_targets(model)
        if not target_modules:
            raise ValueError("Could not detect compatible attention projection modules for LoRA")
        adapter_dir = self.output_dir / "adapter"
        if self.resume_from and (Path(self.resume_from) / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, self.resume_from, is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=target_modules, task_type=TaskType.CAUSAL_LM))
        model.train()
        optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=self.learning_rate)
        losses: list[float] = []
        total_steps = max(1, math.ceil(len(self.dataset) * self.epochs / self.gradient_accumulation))
        step = 0
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        for epoch in range(self.epochs):
            for index, sample in enumerate(self.dataset):
                if self.cancel_check():
                    raise RuntimeError("Training cancelled")
                if step >= self.max_steps:
                    break
                text = self._format(sample)
                tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
                tokens = {key: value.to(self.device) for key, value in tokens.items()}
                output = model(**tokens, labels=tokens["input_ids"])
                loss = output.loss / self.gradient_accumulation
                loss.backward()
                if (index + 1) % self.gradient_accumulation == 0 or index == len(self.dataset) - 1:
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    step += 1
                    value = float(loss.detach().cpu()) * self.gradient_accumulation
                    losses.append(value)
                    pct = min(99.0, step / total_steps * 100)
                    self.progress(pct, f"epoch {epoch + 1}/{self.epochs}, step {step}/{total_steps}, loss {value:.4f}")
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            torch.save({"epoch": epoch + 1, "optimizer": optimizer.state_dict()}, self.output_dir / "training_state.pt")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        result = {"loss": losses, "final_loss": losses[-1] if losses else None, "steps": step, "elapsed_seconds": round(time.time() - started, 2), "adapter": str(adapter_dir)}
        (self.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self.progress(100.0, "training complete")
        return result

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        return device

    @staticmethod
    def _find_lora_targets(model) -> list[str]:
        preferred = {"q_proj", "k_proj", "v_proj", "o_proj", "query_key_value", "q_attn", "c_attn"}
        found = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in preferred}
        return sorted(found)

    @staticmethod
    def _format(sample: dict) -> str:
        return f"Instruction: {sample['instruction']}\nInput: {sample.get('input', '')}\nAnswer: {sample['output']}"
