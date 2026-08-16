from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from huggingface_hub import HfApi

from aether.core.config import settings


@dataclass
class ModelInfo:
    model_id: str
    pipeline_tag: str | None
    downloads: int
    likes: int
    library_name: str | None
    tags: list[str]


class HuggingFaceService:
    def __init__(self) -> None:
        self.api = HfApi(token=settings.hf_token)

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        models = self.api.list_models(search=query, pipeline_tag="text-generation", sort="downloads", limit=limit)
        return [asdict(self._to_info(model)) for model in models]

    def get(self, model_id: str) -> dict[str, Any]:
        model = self.api.model_info(model_id, files_metadata=True)
        data = asdict(self._to_info(model))
        data["siblings"] = [item.rfilename for item in (model.siblings or [])]
        data["sha"] = model.sha
        return data

    def search_datasets(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        datasets = self.api.list_datasets(search=query, sort="downloads", limit=limit)
        return [{"dataset_id": d.id, "downloads": getattr(d, "downloads", 0) or 0, "likes": getattr(d, "likes", 0) or 0, "tags": list(getattr(d, "tags", None) or [])} for d in datasets]

    def dataset_info(self, dataset_id: str) -> dict[str, Any]:
        info = self.api.dataset_info(dataset_id, files_metadata=True)
        return {
            "dataset_id": info.id,
            "sha": info.sha,
            "siblings": [x.rfilename for x in (info.siblings or [])],
            "tags": list(getattr(info, "tags", None) or []),
            "downloads": getattr(info, "downloads", 0) or 0,
            "likes": getattr(info, "likes", 0) or 0,
        }

    @staticmethod
    def _to_info(model: Any) -> ModelInfo:
        return ModelInfo(
            model_id=model.id,
            pipeline_tag=getattr(model, "pipeline_tag", None),
            downloads=getattr(model, "downloads", 0) or 0,
            likes=getattr(model, "likes", 0) or 0,
            library_name=getattr(model, "library_name", None),
            tags=list(getattr(model, "tags", None) or []),
        )
