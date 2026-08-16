from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from aether.core.config import settings
from aether.core.database import connect
from aether.services.dataset import DatasetService
from aether.services.training import SFTTrainer


class JobService:
    def create(self, user_id: int, kind: str, payload: dict) -> str:
        job_id = uuid.uuid4().hex
        with connect() as conn:
            active = conn.execute("SELECT COUNT(*) c FROM jobs WHERE user_id=? AND status IN ('queued','running')", (user_id,)).fetchone()["c"]
            today = conn.execute("SELECT COUNT(*) c FROM jobs WHERE user_id=? AND created_at >= date('now')", (user_id,)).fetchone()["c"]
            if today >= settings.max_jobs_per_day:
                raise ValueError("Daily training job limit reached")
            if active >= settings.max_active_jobs_per_user:
                raise ValueError("Maximum active jobs reached for this user")
            conn.execute("INSERT INTO jobs(id,user_id,kind,status,payload) VALUES(?,?,?,?,?)", (job_id, user_id, kind, "queued", json.dumps(payload)))
        return job_id

    def get(self, job_id: str, user_id: int) -> dict | None:
        with connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id=? AND user_id=?", (job_id, user_id)).fetchone()
        return dict(row) if row else None

    def update(self, job_id: str, **fields) -> None:
        fields["updated_at"] = "CURRENT_TIMESTAMP"
        values = {k: v for k, v in fields.items() if k != "updated_at"}
        assignments = ", ".join(f"{k}=?" for k in values)
        assignments += ", updated_at=CURRENT_TIMESTAMP"
        with connect() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id=?", (*values.values(), job_id))


class JobWorker:
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="aether-job-worker")

    def start(self) -> None:
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            row = self._claim()
            if not row:
                time.sleep(1)
                continue
            try:
                self._run(dict(row))
            except RuntimeError as exc:
                if "cancelled" in str(exc).lower():
                    JobService().update(row["id"], status="cancelled", message="training cancelled")
                else:
                    JobService().update(row["id"], status="failed", error=str(exc), message="job failed")
            except Exception as exc:
                JobService().update(row["id"], status="failed", error=str(exc), message="job failed")

    @staticmethod
    def _claim():
        with connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            conn.execute("UPDATE jobs SET status='running', updated_at=CURRENT_TIMESTAMP WHERE id=? AND status='queued'", (row["id"],))
            return conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()

    def _run(self, row: dict) -> None:
        payload = json.loads(row["payload"])
        if row["kind"] == "train":
            dataset = DatasetService().load(payload["dataset_path"])
            dataset = DatasetService().clean(DatasetService().normalize(dataset, payload.get("mapping")))
            if not dataset:
                raise ValueError("Dataset contains no valid training examples")
            output = settings.data_dir / "users" / str(row["user_id"]) / "runs" / row["id"]
            output.mkdir(parents=True, exist_ok=True)
            JobService().update(row["id"], result_path=str(output / "adapter"))
            resume_from = payload.get("resume_from")
            trainer = SFTTrainer(
                model_id=payload["model_id"], dataset=dataset, output_dir=output,
                learning_rate=payload.get("learning_rate", 2e-4), epochs=payload.get("epochs", 1),
                max_length=min(payload.get("max_length", 512), settings.max_sequence_length),
                gradient_accumulation=max(1, payload.get("gradient_accumulation", 8)),
                device="auto", resume_from=resume_from, max_steps=settings.max_training_steps, cancel_check=lambda: JobService().get(row["id"], row["user_id"])["status"] == "cancelled", progress=lambda p, m: JobService().update(row["id"], progress=p, message=m),
            )
            result = trainer.train()
            JobService().update(row["id"], status="completed", progress=100, message="training complete", result_path=result["adapter"])
        else:
            raise ValueError(f"Unknown job kind: {row['kind']}")


worker = JobWorker()
