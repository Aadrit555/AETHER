from __future__ import annotations

import typer

from aether.core.config import Config
from aether.core.engine import Engine
from aether.core.logger import Logger
from aether.services.huggingface import HuggingFaceService

app = typer.Typer(help="Aether - Beginner-first LLM training platform")


@app.command()
def train(config: str = "configs/qwen.yaml"):
    """Run real supervised LoRA fine-tuning from a YAML configuration."""
    logger = Logger()
    cfg = Config(config)
    Engine(config=cfg, logger=logger).run()


@app.command("search-models")
def search_models(query: str, limit: int = 10):
    """Search live Hugging Face text-generation models."""
    for model in HuggingFaceService().search(query, limit):
        print(f"{model['model_id']} | downloads={model['downloads']} | likes={model['likes']}")


@app.command()
def version():
    print("Aether v0.1.0")


if __name__ == "__main__":
    app()
