from __future__ import annotations

import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from aether.core.config import settings


def evaluate(model_id: str, adapter_path: str, samples: list[dict]) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, token=settings.hf_token).to(device)
    model = PeftModel.from_pretrained(model, adapter_path).eval()
    losses = []
    with torch.inference_mode():
        for sample in samples:
            text = f"Instruction: {sample['instruction']}\nInput: {sample.get('input', '')}\nAnswer: {sample['output']}"
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=settings.max_sequence_length).to(device)
            losses.append(float(model(**tokens, labels=tokens["input_ids"]).loss.cpu()))
    mean_loss = sum(losses) / len(losses) if losses else None
    return {"examples": len(losses), "loss": mean_loss, "perplexity": math.exp(mean_loss) if mean_loss is not None and mean_loss < 20 else None}


def grounding(context: str, answer: str) -> dict:
    from transformers import AutoModelForSequenceClassification
    model_id = "cross-encoder/nli-MiniLM2-L6-H768"
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.hf_token)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=settings.hf_token)
    inputs = tokenizer(context, answer, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0]
    labels = [str(x).lower() for x in getattr(model.config, "id2label", {}).values()]
    if len(labels) != len(probs):
        labels = ["contradiction", "neutral", "entailment"][: len(probs)]
    scores = {labels[i]: float(probs[i]) for i in range(len(probs))}
    best = max(scores, key=scores.get)
    return {"label": best, "scores": scores, "model": model_id}
