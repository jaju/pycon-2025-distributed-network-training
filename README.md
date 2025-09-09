# Distributed Training Workshop (Participants)

Welcome. This is the one‑page guide for the session. It focuses on fast setup, clear commands, and consistent outputs.

## What You’ll Learn

- Build intuition by evolving a baseline → manual distributed → DDP → FSDP.
- Track core signals: throughput, speedup vs. baseline, and correctness via distributed validation.
- Use lightweight instrumentation and TensorBoard to compare runs.

## Prerequisites

**Windows Users:** Install the latest stable version of WSL (Windows Subsystem for Linux) first:
```powershell
wsl --install
```
Restart your computer, then follow the Linux instructions below in your WSL terminal.

- Python 3.12+
- uv (dependency manager)
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows (WSL): `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Or: `pipx install uv`
- just (task runner)
  - macOS: `brew install just`
  - Ubuntu/Debian: `apt install just`
  - Fedora: `dnf install just`
  - Windows (WSL): Use the appropriate Linux command above for your WSL distro
  - Fallback (any Linux): `cargo install just`

## Setup (one time)

1) Install deps: `just setup`
2) Data download: `just download-data` (CIFAR‑10). Do this *BEFORE* coming to the workshop so we do not choke the WiFi.
3) Quick check: `uv run python -c "import torch; print('torch', torch.__version__)"`
4) Verify `uv` is on PATH: `uv --version`

## Start TensorBoard (optional)

- `just tb` → open http://localhost:6006

## Repo Layout (what you’ll use)

- `teach/`: minimal, one‑screen teaching scripts (run single process or echo two commands)
- `reference/`: production‑leaning templates with richer metrics (echo two commands for distributed)
- Shared primitives: `reference/config.py` (unified TrainingConfig), `reference/metrics.py` (MetricsMonitor)

## Environment (.env)

`just` reads `.env` and includes values in echoed commands. Create/edit `.env` to prefill:

```
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
BACKEND=gloo       # for CPU runs; use nccl on CUDA
DEVICE=cpu         # or cuda, mps
GLOO_SOCKET_IFNAME=lo   # Linux loopback (lo0 on macOS)
# NCCL_SOCKET_IFNAME=eth0  # example for CUDA + NCCL on Ethernet
```

## How to Run — TL;DR

- Step 0 (teach): `just step0-basic --epochs 1 --scheduler none`
- Step 0 (reference): `just step0-basic-reference --epochs 1 --scheduler none`

- Step 1 (manual dist; echo two commands):
  - Teach: `just step1-dist`
  - Reference: `just step1-dist-reference`

- Step 2 (DDP; echo two commands):
  - Teach: `just step2-ddp`
  - Reference: `just step2-ddp-reference`

- Step 3 (FSDP):
  - Teach simulate (single process): `just step3-fsdp --epochs 1 --device cpu --scheduler none`
  - Reference (echo two commands): `just step3-fsdp-reference`

Copy/paste the echoed commands into two terminals exactly as shown. Rank 0 prints and writes JSON/plots/TB.

## Step Details

### Step 0 — Baseline (single process)
- ResNet‑18 on CIFAR‑10 (or synthetic fallback), clean loop, deterministic setup.
- Expect one concise epoch summary; artifacts in `runs/` and `artifacts/`.

### Step 1 — Manual Distributed (two terminals)
- Naive per‑parameter all‑reduce after backward; no bucketing or overlap.
- Echo commands:
  - Teach: `just step1-dist`
  - Reference: `just step1-dist-reference`
- Expect comm_overhead and num_allreduces; proper global validation reductions.

### Step 2 — DDP (two terminals)
- Bucketing and overlap for efficient gradient sync.
- Echo commands:
  - Teach: `just step2-ddp`
  - Reference: `just step2-ddp-reference`
- Expect improved throughput vs Step 1 and speedup/efficiency vs a baseline.

### Step 3 — FSDP
- Teach simulate (single process): proxies only; verifies the mental model.
- Reference (two terminals): real sharding (CPU/gloo or CUDA/NCCL) with structured metrics.

## Data

- CIFAR‑10 (optional): `just download-data` (defaults to `./data`).
- Synthetic data is on by default in teaching scripts and reference quick runs for fast iteration.

## Troubleshooting (quick)

- Interface binding (gloo): set `GLOO_SOCKET_IFNAME` (Linux: `lo`, macOS: `lo0`) in `.env`.
- Port conflicts: change `MASTER_PORT` (e.g., 29501) in `.env` and rerun the echoed commands.
- CPU vs CUDA: set `DEVICE=cuda` and `BACKEND=nccl` for NVIDIA GPUs; otherwise stick to CPU/gloo.
- TensorBoard: ensure at least one epoch completed; logs land under `runs/`.
 - Apple Silicon (MPS): DDP collectives over gloo are not supported; use `--device cpu` for distributed runs, or run Step 0 on `mps` for single‑process speed.

## What “good” looks like

- Step 0: Clean loss curve; metrics saved.
- Step 1: Global validation metrics aggregated correctly; comm stats present.
- Step 2: Higher throughput vs Step 1; speedup/efficiency computed vs baseline (magnitude depends on model/network/host).
- Step 3: FSDP proxies (teach) or actual sharding performance (reference) with clear summaries.

## After the Workshop

- Increase epochs, adjust batch size, or try schedulers (`--scheduler cosine|step|onecycle`).
- Compare runs in TensorBoard; validate differences with global aggregates and JSON.
- Explore the reference templates and port patterns into your codebase.
