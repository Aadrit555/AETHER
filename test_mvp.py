from __future__ import annotations

import tempfile
from pathlib import Path

from aether.services.dataset import DatasetService


def main() -> None:
    service = DatasetService()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.jsonl"
        path.write_text('{"instruction":"A","output":"B"}\n{"instruction":"A","output":"B"}\n{"instruction":"","output":"bad"}\n', encoding="utf-8")
        rows = service.load(path)
        report = service.validate(rows)
        assert report.total == 3
        assert report.valid == 2
        assert report.invalid == 1
        assert report.duplicates == 1
        assert len(service.clean(rows)) == 1
    print("dataset smoke test: ok")


if __name__ == "__main__":
    main()
