from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aether.core.config import settings
from aether.core.database import init_db, connect
from aether.services.auth import authenticate, create_user, token_for, user_from_token
from aether.services.dataset import DatasetService
from aether.services.huggingface import HuggingFaceService
from aether.services.jobs import JobService, worker
from aether.services.inference import generate
from aether.services.evaluation import evaluate, grounding
from aether.services.model_loader import ModelLoader

VERSION = "0.3.0"
app = FastAPI(title="Aether", version=VERSION, docs_url="/docs", redoc_url="/redoc")
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])
hf = HuggingFaceService()
loader = ModelLoader()
datasets = DatasetService()
jobs = JobService()


class ModelSearch(BaseModel):
    q: str = Field(min_length=1, max_length=200)
    limit: int = Field(default=20, ge=1, le=100)


class DatasetSearch(ModelSearch):
    pass


class AuthRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=128)




class GroundingRequest(BaseModel):
    context: str = Field(min_length=1, max_length=20000)
    answer: str = Field(min_length=1, max_length=10000)


class EvaluationRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    adapter_job_id: str
    dataset_id: str


class InferenceRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    prompt: str = Field(min_length=1, max_length=10000)
    adapter_job_id: str | None = None
    max_new_tokens: int = Field(default=128, ge=1, le=512)


class TrainRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=200)
    dataset_id: str | None = None
    dataset_path: str | None = None
    mapping: dict[str, str] | None = None
    epochs: int = Field(default=1, ge=1, le=10)
    learning_rate: float = Field(default=2e-4, gt=0, le=1e-2)
    max_length: int = Field(default=512, ge=64, le=settings.max_sequence_length)
    gradient_accumulation: int = Field(default=8, ge=1, le=128)


def current_user(authorization: str | None = Header(default=None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return user_from_token(authorization[7:])
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Invalid or expired token") from exc


@app.on_event("startup")
def startup() -> None:
    if settings.environment == "production" and settings.jwt_secret == "change-me-in-production":
        raise RuntimeError("AETHER_JWT_SECRET must be changed in production")
    init_db()
    if os.getenv("AETHER_WORKER_MODE", "embedded") == "embedded":
        worker.start()


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "aether-api", "version": VERSION, "environment": settings.environment}


@app.get("/ready")
def ready() -> dict[str, Any]:
    with connect() as conn:
        conn.execute("SELECT 1")
    return {"status": "ready", "service": "aether-api", "version": VERSION}


@app.post("/auth/register")
def register(request: AuthRequest):
    try:
        user_id = create_user(request.email, request.password)
    except Exception as exc:
        raise HTTPException(status_code=409, detail="Email already registered") from exc
    return {"access_token": token_for(user_id), "token_type": "bearer"}


@app.post("/auth/login")
def login(request: AuthRequest):
    user_id = authenticate(request.email, request.password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    return {"access_token": token_for(user_id), "token_type": "bearer"}


@app.post("/models/search")
def search_models(request: ModelSearch):
    try:
        return {"models": hf.search(request.q, request.limit)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/models/{model_id:path}")
def model_info(model_id: str):
    try:
        remote = hf.get(model_id)
        local = loader.inspect(model_id)
        if local.get("parameter_count_millions", 0) > settings.max_model_params_millions:
            local["training_allowed"] = False
            local["training_reason"] = f"Model exceeds configured {settings.max_model_params_millions}M parameter limit"
        else:
            local["training_allowed"] = True
        return {**remote, **local}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/datasets/search")
def search_datasets(request: DatasetSearch):
    try:
        return {"datasets": hf.search_datasets(request.q, request.limit)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/datasets/hub/{dataset_id:path}")
def dataset_info(dataset_id: str):
    try:
        return hf.dataset_info(dataset_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), user_id: int = Depends(current_user)):
    suffix = Path(file.filename or "dataset.jsonl").suffix.lower()
    if suffix not in {".json", ".jsonl", ".csv"}:
        raise HTTPException(status_code=400, detail="Only JSON, JSONL and CSV are supported")
    user_dir = settings.data_dir / "users" / str(user_id) / "datasets"
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > settings.max_upload_mb * 1024 * 1024:
                path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="Dataset exceeds upload limit")
            out.write(chunk)
    try:
        raw = datasets.load(path)
        normalized = datasets.normalize(raw)
        report = datasets.validate(normalized)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset_id = uuid.uuid4().hex
    with connect() as conn:
        conn.execute("INSERT INTO datasets(id,user_id,name,path,source,rows,valid_rows,invalid_rows,duplicate_rows) VALUES(?,?,?,?,?,?,?,?,?)", (dataset_id, file.filename, str(path), "upload", report.total, report.valid, report.invalid, report.duplicates))
    return {"dataset_id": dataset_id, "name": file.filename, "report": report.__dict__}


@app.post("/datasets/from-hub")
def import_hub_dataset(dataset_id: str, user_id: int = Depends(current_user)):
    from datasets import load_dataset
    import uuid
    user_dir = settings.data_dir / "users" / str(user_id) / "datasets"
    user_dir.mkdir(parents=True, exist_ok=True)
    try:
        ds = load_dataset(dataset_id, split="train", token=settings.hf_token, streaming=False)
        limit = min(len(ds), 10000)
        rows = [dict(ds[i]) for i in range(limit)]
        normalized = datasets.normalize(rows)
        report = datasets.validate(normalized)
        path = user_dir / f"{uuid.uuid4().hex}.jsonl"
        cleaned = datasets.clean(normalized)
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cleaned), encoding="utf-8")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load dataset: {exc}") from exc
    dataset_key = uuid.uuid4().hex
    with connect() as conn:
        conn.execute("INSERT INTO datasets(id,user_id,name,path,source,rows,valid_rows,invalid_rows,duplicate_rows) VALUES(?,?,?,?,?,?,?,?,?)", (dataset_key, dataset_id, str(path), "huggingface", report.total, report.valid, report.invalid, report.duplicates))
    return {"dataset_id": dataset_key, "dataset_name": dataset_id, "report": report.__dict__, "clean_rows": len(cleaned)}


@app.post("/training/start")
def start_training(request: TrainRequest, user_id: int = Depends(current_user)):
    if not request.dataset_path and not request.dataset_id:
        raise HTTPException(status_code=400, detail="Provide dataset_id")
    if request.dataset_id:
        with connect() as conn:
            row = conn.execute("SELECT path FROM datasets WHERE id=? AND user_id=?", (request.dataset_id, user_id)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Dataset not found")
        dataset_path = row["path"]
    else:
        raise HTTPException(status_code=400, detail="Direct dataset paths are not accepted")
    info = loader.inspect(request.model_id)
    if info["parameter_count_millions"] > settings.max_model_params_millions:
        raise HTTPException(status_code=400, detail="Model exceeds configured training limit")
    payload = request.model_dump(exclude_none=True) | {"dataset_path": dataset_path}
    try:
        job_id = jobs.create(user_id, "train", payload)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {"job_id": job_id, "status": "queued"}


@app.post("/training/{job_id}/resume")
def resume_training(job_id: str, user_id: int = Depends(current_user)):
    job = jobs.get(job_id, user_id)
    if not job or job["status"] not in {"failed", "cancelled", "completed"}:
        raise HTTPException(status_code=409, detail="Job cannot be resumed")
    if not job.get("result_path"):
        raise HTTPException(status_code=409, detail="No checkpoint exists for this job")
    payload = json.loads(job["payload"])
    payload["resume_from"] = job["result_path"]
    try:
        new_id = jobs.create(user_id, "train", payload)
    except ValueError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return {"job_id": new_id, "status": "queued", "resumed_from": job["result_path"]}


@app.get("/training/{job_id}")
def training_status(job_id: str, user_id: int = Depends(current_user)):
    job = jobs.get(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job["payload"] = json.loads(job["payload"])
    return job


@app.post("/inference")
def inference(request: InferenceRequest, user_id: int = Depends(current_user)):
    try:
        adapter_path = None
        if request.adapter_job_id:
            job = jobs.get(request.adapter_job_id, user_id)
            if not job or job.get("status") != "completed":
                raise HTTPException(status_code=404, detail="Completed training job not found")
            adapter_path = job.get("result_path")
        answer = generate(request.model_id, adapter_path, request.prompt, request.max_new_tokens)
        return {"answer": answer}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/my/models")
def my_models(user_id: int = Depends(current_user)):
    with connect() as conn:
        rows = conn.execute("SELECT id, kind, status, result_path, message, created_at, updated_at FROM jobs WHERE user_id=? AND status='completed' ORDER BY created_at DESC", (user_id,)).fetchall()
    return {"models": [dict(row) for row in rows]}


@app.post("/training/{job_id}/cancel")
def cancel_training(job_id: str, user_id: int = Depends(current_user)):
    job = jobs.get(job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="Job is not active")
    jobs.update(job_id, status="cancelled", message="cancellation requested")
    return {"status": "cancelled"}


@app.post("/evaluation/grounding")
def grounding_check(request: GroundingRequest, user_id: int = Depends(current_user)):
    try:
        return grounding(request.context, request.answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/evaluation")
def evaluation(request: EvaluationRequest, user_id: int = Depends(current_user)):
    job = jobs.get(request.adapter_job_id, user_id)
    if not job or job["status"] != "completed" or not job.get("result_path"):
        raise HTTPException(status_code=404, detail="Completed training job not found")
    with connect() as conn:
        row = conn.execute("SELECT path FROM datasets WHERE id=? AND user_id=?", (request.dataset_id, user_id)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    raw = datasets.load(row["path"])
    samples = datasets.clean(datasets.normalize(raw))
    try:
        return evaluate(request.model_id, job["result_path"], samples)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
