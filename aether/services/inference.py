from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from aether.core.config import settings


def generate(model_id: str, adapter_path: str | None, prompt: str, max_new_tokens: int = 128) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=settings.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, token=settings.hf_token).to(device)
    if adapter_path:
        if not Path(adapter_path).exists():
            raise FileNotFoundError("Adapter not found")
        model = PeftModel.from_pretrained(model, adapter_path)
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=settings.max_sequence_length).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=min(max_new_tokens, 512), do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
