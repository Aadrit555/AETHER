# Aether

Aether is a beginner-first LLM training workspace. It lets users discover models and datasets on Hugging Face, upload their own datasets, validate/clean them, fine-tune compatible causal language models with LoRA, track real jobs, evaluate checkpoints, test models, and keep trained adapters.

Aether does not depend on LlamaFactory and does not ship a copied LlamaFactory UI.

## Local development

### Windows + NVIDIA

Create environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install a CUDA-enabled PyTorch build first if your machine has NVIDIA GPU. Example for CUDA 12.8 wheels:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Run API:

```powershell
uvicorn aether.api.server:app --reload
```

Run frontend in another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Production Docker deployment

Copy environment file:

```bash
cp .env.example .env
```

Set a long random `AETHER_JWT_SECRET`, your public `AETHER_DOMAIN`, and optional `HF_TOKEN`.

For a GPU host with Docker + NVIDIA Container Toolkit:

```bash
docker compose up -d --build
```

Caddy terminates HTTPS automatically when `AETHER_DOMAIN` points at the server.

Architecture:

```text
Browser
  -> Caddy
  -> React/Nginx
  -> FastAPI
  -> SQLite + persistent storage
  -> GPU worker
  -> Hugging Face / PyTorch / PEFT
```

API and worker are separate containers. Only worker receives GPU access.

## Free public GPU

Aether is designed around a replaceable GPU worker. The included Docker deployment works on any GPU host with NVIDIA Container Toolkit. A free GPU provider cannot be guaranteed: quotas and availability are controlled by provider. Hugging Face ZeroGPU is an optional provider target, but ZeroGPU Spaces use their own Space runtime rather than this Docker Compose stack. Do not promise unlimited free public training.

For a genuinely free public launch, keep strict limits:

- small compatible models
- LoRA only
- dataset upload limit
- maximum sequence length
- maximum training steps
- one active job per user
- daily job quota
- queued jobs when GPU is busy

## Public deployment checklist

- Use HTTPS through Caddy.
- Replace `AETHER_JWT_SECRET`.
- Set `AETHER_CORS_ORIGINS` to your real frontend origin.
- Keep GPU worker isolated from public HTTP traffic.
- Keep model/data volumes persistent.
- Set quotas appropriate to available GPU budget.
- Back up `aether-data`.
- Do not expose SQLite files or model-cache paths through web server.
- Add external monitoring before opening high traffic.

## Current product flow

```text
Find model -> Find dataset -> Validate -> Clean -> Train -> Track -> Evaluate -> Test -> Keep adapter
```

Supported user dataset files: JSON, JSONL, CSV.

Supported model source: Hugging Face Hub and local model loading in the core engine. Public UI currently exposes Hugging Face discovery plus uploaded datasets.
