from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from aether.core.config import settings


@dataclass
class LoadedModel:
    model_id: str
    architecture: list[str]
    parameter_count: int
    device: str
    dtype: str
    tokenizer_loaded: bool


class ModelLoader:
    def inspect(self, model_id: str) -> dict:
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=False, token=settings.hf_token)
        architecture = list(getattr(config, "architectures", None) or [])
        parameter_count = self._estimate_parameters(config)
        return {
            "model_id": model_id,
            "architecture": architecture,
            "parameter_count": parameter_count,
            "parameter_count_millions": round(parameter_count / 1_000_000, 2),
            "model_type": getattr(config, "model_type", None),
            "torch_dtype": str(getattr(config, "torch_dtype", None)),
        }

    def load(self, model_id: str, device: str = "auto") -> tuple[AutoModelForCausalLM, AutoTokenizer, LoadedModel]:
        resolved_device = self._resolve_device(device)
        dtype = torch.float16 if resolved_device == "cuda" else torch.float32
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, token=settings.hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=False, token=settings.hf_token)
        model.to(resolved_device)
        metadata = LoadedModel(
            model_id=model_id,
            architecture=list(getattr(model.config, "architectures", None) or []),
            parameter_count=sum(parameter.numel() for parameter in model.parameters()),
            device=resolved_device,
            dtype=str(dtype),
            tokenizer_loaded=True,
        )
        return model, tokenizer, metadata

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        return device

    @staticmethod
    def _estimate_parameters(config) -> int:
        vocab = getattr(config, "vocab_size", 0) or 0
        hidden = getattr(config, "hidden_size", 0) or 0
        layers = getattr(config, "num_hidden_layers", 0) or 0
        intermediate = getattr(config, "intermediate_size", 0) or 0
        if not all((vocab, hidden, layers, intermediate)):
            return 0
        # Conservative architectural estimate. Exact count is available after weights load.
        embedding = vocab * hidden
        attention_ffn = layers * (4 * hidden * hidden + 2 * hidden * intermediate)
        return int(embedding + attention_ffn + embedding)
