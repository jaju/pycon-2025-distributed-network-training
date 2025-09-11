set dotenv-load
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

MPLCONFIGDIR := "./.mplcache"
PY := "uv run python"

# Install dependencies using uv (shared env across steps)
setup:
    uv sync --all-groups

# TensorBoard helpers
tb:
    MPLCONFIGDIR={{MPLCONFIGDIR}} uv run tensorboard --logdir runs --port 6006

# Lint and format
lint:
    uv run ruff check .

fmt:
    uv run ruff format .

# Types (non-gating guidance)
types:
    uv run pyrefly check . || true

# Step 0: Basic (teaching) and reference
step0-basic *ARGS:
    MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m teach.step0_simple {{ARGS}}

step0-basic-reference *ARGS:
    MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m reference.simple_train --scheduler ${SCHEDULER:-cosine} {{ARGS}}

# Step 1: Manual dist (teaching) and reference
step1-dist *ARGS:
    @echo "# Terminal 1 (master)"; \
    echo "MPLCONFIGDIR={{MPLCONFIGDIR}} RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29550} {{PY}} -m teach.step1_simple_dist {{ARGS}}"; \
    echo ""; \
    echo "# Terminal 2 (worker)"; \
    echo "MPLCONFIGDIR={{MPLCONFIGDIR}} RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29550} {{PY}} -m teach.step1_simple_dist {{ARGS}}"

step1-dist-reference *ARGS:
    @echo "# Terminal 1 (process 0)"; \
    echo "${GLOO_SOCKET_IFNAME:+GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} }BACKEND=${BACKEND:-gloo} MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29510} -m reference.simple_dist_train {{ARGS}}"; \
    echo ""; \
    echo "# Terminal 2 (process 1)"; \
    echo "${GLOO_SOCKET_IFNAME:+GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} }BACKEND=${BACKEND:-gloo} MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29510} -m reference.simple_dist_train {{ARGS}}"

# Step 2: DDP (teaching) and reference
step2-ddp *ARGS:
    @echo "# Terminal 1 (master)"; \
    echo "MPLCONFIGDIR={{MPLCONFIGDIR}} BACKEND=${BACKEND:-gloo} RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29560} {{PY}} -m teach.step2_ddp {{ARGS}}"; \
    echo ""; \
    echo "# Terminal 2 (worker)"; \
    echo "MPLCONFIGDIR={{MPLCONFIGDIR}} BACKEND=${BACKEND:-gloo} RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} MASTER_PORT=${MASTER_PORT:-29560} {{PY}} -m teach.step2_ddp {{ARGS}}"

step2-ddp-reference *ARGS:
    @echo "# Terminal 1 (process 0)"; \
    echo "${GLOO_SOCKET_IFNAME:+GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} }BACKEND=${BACKEND:-gloo} MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29520} -m reference.ddp_train {{ARGS}}"; \
    echo ""; \
    echo "# Terminal 2 (process 1)"; \
    echo "${GLOO_SOCKET_IFNAME:+GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} }BACKEND=${BACKEND:-gloo} MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29520} -m reference.ddp_train {{ARGS}}"

# Step 3: FSDP (teaching simulate) and reference
step3-fsdp *ARGS:
    MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m teach.step3_fsdp_sim {{ARGS}}

step3-fsdp-reference *ARGS:
    @echo "# Terminal 1 (process 0)"; \
    echo "${GLOO_SOCKET_IFNAME:+GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} }BACKEND=${BACKEND:-nccl} MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=0 --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29530} -m reference.fsdp_train --device ${DEVICE:-cpu} {{ARGS}}"; \
    echo ""; \
    echo "# Terminal 2 (process 1)"; \
    echo "${GLOO_SOCKET_IFNAME:+GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} }BACKEND=${BACKEND:-nccl} MPLCONFIGDIR={{MPLCONFIGDIR}} {{PY}} -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=1 --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-29530} -m reference.fsdp_train --device ${DEVICE:-cpu} {{ARGS}}"

# Download CIFAR-10 once into DATA_DIR (default: ./data)
download-data:
    #!/usr/bin/env bash
    set -euo pipefail
    DATA_DIR="${DATA_DIR:-data}"
    mkdir -p "${DATA_DIR}"
    {{PY}} - <<'PY'
    import os, sys
    try:
        from torchvision import datasets  # type: ignore
    except Exception as e:
        sys.stderr.write("torchvision not available. Run `just setup` first.\n")
        raise
    data_dir = os.environ.get("DATA_DIR", "data")
    datasets.CIFAR10(root=data_dir, train=True, download=True)
    datasets.CIFAR10(root=data_dir, train=False, download=True)
    print(f"Downloaded CIFAR-10 to {data_dir}")
    PY


# Render delivery slide (Markdown + SVG diagrams) using Marp CLI
diagrams:
    mkdir -p artifacts/diagrams
    mmdc -i docs/diagrams/data-parallel.mmd   -o artifacts/diagrams/data-parallel.svg   -b transparent -w 1600 -H 900
    mmdc -i docs/diagrams/ring-allreduce.mmd  -o artifacts/diagrams/ring-allreduce.svg  -b transparent -w 1600 -H 900
    mmdc -i docs/diagrams/overlap-gantt.mmd   -o artifacts/diagrams/overlap-gantt.svg   -b transparent -w 1600 -H 900
    mmdc -i docs/diagrams/fsdp-sharding.mmd   -o artifacts/diagrams/fsdp-sharding.svg   -b transparent -w 1600 -H 900

slide-pdf:
    just diagrams
    mkdir -p artifacts
    (cd docs && marp --allow-local-files --pdf -o ../artifacts/delivery-slide.pdf delivery-slide.md)

slide-serve:
    just diagrams
    (cd docs && marp --allow-local-files --server delivery-slide.md)

slide:
    just slide-pdf
