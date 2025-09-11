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

1) Copy the appropriate `.env.<os>` to `.env` (e.g., `.env.macos` or `.env.linux`).
2) Verify `uv` is on PATH: `uv --version`
3) Install deps: `just setup`
4) Activate the Python virtual environment: `source .venv/bin/activate`
5) Data download: `just download-data` (CIFAR‑10). Do this *BEFORE* coming to the workshop so we do not choke the WiFi.
6) Quick check: `uv run python -c "import torch; print('torch', torch.__version__)"`

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

## Data

- CIFAR‑10 (optional): `just download-data` (defaults to `./data`).
- Synthetic data is on by default in teaching scripts and reference quick runs for fast iteration.

### About the Dataset

We use the CIFAR‑10 dataset for this workshop.

CIFAR-10 and CIFAR-100 are standard benchmark datasets for image classification. They’re small, diverse, and great for trying out architectures like ResNet.

| Dataset | #Images | Image Size | #Classes | Train / Test split | Extra structure |
|---|---|---|---|---|---|
| **CIFAR-10** | 60,000 total | 32×32 colour (RGB) | 10 classes | 50,000 train / 10,000 test | Uniform classes; handles e.g. airplane, car, bird, cat, deer, dog, frog, horse, ship, truck ([cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com)) |
| **CIFAR-100** | 60,000 total | 32×32 colour (RGB) | 100 classes | 50,000 train / 10,000 test | Each image has a “fine” class (100) and a “coarse superclass” (20) grouping ([cs.toronto.edu](https://www.cs.toronto.edu/~kriz/cifar.html?utm_source=chatgpt.com)) |


### 🎯 Example Classes

CIFAR-10 classes include:
> `airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck` ([geeksforgeeks.org](https://www.geeksforgeeks.org/deep-learning/cifar-10-image-classification-in-tensorflow/))


## How to Run — TL;DR

**NOTE**: Do NOT run via vanilla `python` or `torchrun` directly. Use `just` to ensure `.env` is read and env vars are set consistently.

### Step 0 - Baseline, Single Process
Basic: `just step0-basic`

Reference (with example extra knob-arguments): `just step0-basic-reference --epochs 2 --scheduler cosine`

### Step 1 (manual dist)
Basic: `just step1-dist`

Reference (echo two commands): `just step1-dist-reference`

Example output snippet:
```shell
# Terminal 1 (process 0)
GLOO_SOCKET_IFNAME=lo0 BACKEND=gloo MPLCONFIGDIR=./.mplcache uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 -m reference.simple_dist_train

# Terminal 2 (process 1)
GLOO_SOCKET_IFNAME=lo0 BACKEND=gloo MPLCONFIGDIR=./.mplcache uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 --master_port=29500 -m reference.simple_dist_train
```

### Step 2 (DDP)

Basic: `just step2-ddp`

Reference: `just step2-ddp-reference`

Example output snippet:
```shell
# Terminal 1 (process 0)
GLOO_SOCKET_IFNAME=lo0 BACKEND=gloo MPLCONFIGDIR=./.mplcache uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 -m reference.ddp_train

# Terminal 2 (process 1)
GLOO_SOCKET_IFNAME=lo0 BACKEND=gloo MPLCONFIGDIR=./.mplcache uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 --master_port=29500 -m reference.ddp_train
```

### Step 3 (FSDP)

Reference: `just step3-fsdp-reference --device cuda`

Example output snippet:
```shell
# Terminal 1 (process 0)
GLOO_SOCKET_IFNAME=lo0 BACKEND=nccl MPLCONFIGDIR=./.mplcache uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 -m reference.fsdp_train --device cpu --device cuda

# Terminal 2 (process 1)
GLOO_SOCKET_IFNAME=lo0 BACKEND=nccl MPLCONFIGDIR=./.mplcache uv run python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=127.0.0.1 --master_port=29500 -m reference.fsdp_train --device cpu --device cuda
```

Copy/paste the echoed commands into two terminals exactly as shown. Rank 0 prints and writes JSON/plots/TB.

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
- Step 3: FSDP - sharding performance (reference-only) with clear summaries.

## After the Workshop

- Increase epochs, adjust batch size, or try schedulers (`--scheduler cosine|step|onecycle`).
- Compare runs in TensorBoard; validate differences with global aggregates and JSON.
- Explore the reference templates and port patterns into your codebase.
