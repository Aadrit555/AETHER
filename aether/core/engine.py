from __future__ import annotations

from datetime import datetime
from pathlib import Path

from aether.services.dataset import DatasetService
from aether.services.model_loader import ModelLoader
from aether.services.training import SFTTrainer


class Engine:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def run(self):
        self.logger.banner()
        self.logger.info("Loading configuration...")
        self.logger.configuration(self.config)

        model_name = self.config.model["name"]
        dataset_path = self.config.dataset["path"]
        output_root = Path(self.config.get("output_dir", "outputs"))
        output_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")

        self.logger.info(f"Loading model: {model_name}")
        loader = ModelLoader()
        model, tokenizer, metadata = loader.load(
            model_name,
            device=self.config.get("device", "auto"),
        )
        self.logger.success(
            f"Loaded {metadata.model_id} ({metadata.parameter_count:,} parameters) on {metadata.device}."
        )

        self.logger.info(f"Loading dataset: {dataset_path}")
        dataset_service = DatasetService()
        raw = dataset_service.load(dataset_path)
        report = dataset_service.validate(raw)
        dataset = dataset_service.clean(raw)
        self.logger.success(
            f"Dataset: {report.total} total, {report.valid} valid, {report.invalid} invalid, {report.duplicates} duplicates; {len(dataset)} kept."
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            output_dir=output_dir,
            learning_rate=float(self.config.training.get("learning_rate", 2e-4)),
            epochs=int(self.config.training.get("epochs", 1)),
            device=self.config.get("device", "auto"),
        )
        self.logger.info("Starting real LoRA fine-tuning...")
        losses = trainer.train()
        self.logger.success(f"Training complete. Final loss: {losses[-1]:.4f}")
        self.logger.success(f"Adapter saved to: {output_dir / 'adapter'}")
