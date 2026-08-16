from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class DatasetReport:
    total: int
    valid: int
    invalid: int
    duplicates: int


class DatasetService:
    REQUIRED = ("instruction", "output")

    def load(self, path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows: list[dict[str, Any]] = []
            errors: list[str] = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("row is not an object")
                    rows.append(value)
                except Exception as exc:
                    errors.append(f"line {number}: {exc}")
            if errors:
                raise ValueError("Invalid JSONL rows: " + "; ".join(errors[:10]))
            return rows
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            data = data["data"]
        if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
            raise ValueError("Dataset JSON must contain a list of objects")
        return data

    def normalize(self, samples: list[dict[str, Any]], mapping: dict[str, str] | None = None) -> list[dict[str, Any]]:
        mapping = mapping or {}
        out = []
        for sample in samples:
            instruction_key = mapping.get("instruction") or self._first(sample, ("instruction", "prompt", "question", "input"))
            output_key = mapping.get("output") or self._first(sample, ("output", "response", "answer", "completion"))
            input_key = mapping.get("input") or self._first(sample, ("input", "context"))
            out.append({"instruction": str(sample.get(instruction_key, "")), "input": str(sample.get(input_key, "")) if input_key else "", "output": str(sample.get(output_key, ""))})
        return out

    def validate(self, samples: list[dict[str, Any]]) -> DatasetReport:
        valid = invalid = duplicates = 0
        seen: set[str] = set()
        for sample in samples:
            key = json.dumps(sample, sort_keys=True, ensure_ascii=False)
            if key in seen:
                duplicates += 1
            seen.add(key)
            if all(isinstance(sample.get(field), str) and sample[field].strip() for field in self.REQUIRED):
                valid += 1
            else:
                invalid += 1
        return DatasetReport(len(samples), valid, invalid, duplicates)

    def clean(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for sample in samples:
            normalized = {"instruction": str(sample.get("instruction", "")).strip(), "input": str(sample.get("input", "")).strip(), "output": str(sample.get("output", "")).strip()}
            if not normalized["instruction"] or not normalized["output"]:
                continue
            key = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            output.append(normalized)
        return output

    @staticmethod
    def split(samples: list[dict[str, Any]], validation_ratio: float = 0.1, seed: int = 42) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        items = list(samples)
        random.Random(seed).shuffle(items)
        cut = max(1, int(len(items) * validation_ratio)) if len(items) > 1 else 0
        return items[cut:], items[:cut]

    @staticmethod
    def _first(sample: dict[str, Any], keys: tuple[str, ...]) -> str | None:
        return next((key for key in keys if key in sample), None)
