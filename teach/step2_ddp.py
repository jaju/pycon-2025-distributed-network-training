"""
Step 2 (Teaching): Minimal DDP training on CIFAR-10.

Requirements:
- Run with at least two processes (set RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
- Uses torch.nn.parallel.DistributedDataParallel (DDP)
- CIFAR-10 if available in DATA_DIR (no auto-download); synthetic fallback otherwise
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

from teach.common import device_str, init_dist, load_dataset


def prepare_data(batch_size: int, rank: int, world: int):
    from torch.utils.data import DataLoader
    import torch.utils.data as data

    data_dir = os.getenv("DATA_DIR", "data")
    train_ds, val_ds = load_dataset(data_dir)

    # DISTRIBUTED SAMPLER CHALLENGE:
    #
    # In Step 1, you manually sharded data with Subset() - crude but functional.
    # DDP needs proper data distribution that handles uneven dataset sizes,
    # controls shuffling per epoch, and ensures each rank sees different data.
    #
    # Problem: How do we automatically and fairly distribute data across ranks?
    #
    # Your task: Create DistributedSampler instances for train and validation sets.
    #
    # Hint 1: Use torch.utils.data.distributed.DistributedSampler
    # Hint 2: Pass num_replicas=world, rank=rank to the sampler so it knows the setup
    # Hint 3: Use shuffle=True for training (randomize each epoch), shuffle=False for validation
    # Hint 4: DistributedSampler automatically handles remainder samples when dataset
    #         size doesn't divide evenly by world_size
    #
    # Key insight: This replaces manual Subset() slicing from Step 1 with proper load balancing!

    # TODO: Implement this section

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def prepare_model_ddp(dev: str, world_size: int, rank: int):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torchvision import models

    # Optional: use LOCAL_RANK for CUDA device placement if provided
    if dev == "cuda":
        idx = int(os.getenv("LOCAL_RANK", "0"))
        torch.cuda.set_device(idx)
        device = torch.device("cuda", idx)
    else:
        device = torch.device("cpu")

    model = models.resnet18(weights=None, num_classes=10).to(device)

    # Synchronize model weights across all ranks before DDP wrapping
    if world_size > 1:
        if rank == 0:
            print("Broadcasting initial weights from rank 0...", flush=True)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        if rank == 0:
            print("Weight broadcast complete.", flush=True)

    # DDP MODEL WRAPPING CHALLENGE:
    #
    # In Step 1, you manually synchronized gradients with allreduce_grads().
    # DDP automates this by wrapping your model and intercepting backward() calls.
    #
    # Problem: How do we convert a regular PyTorch model into a distributed one?
    #
    # Your task: Wrap the model with DistributedDataParallel to enable automatic
    # gradient synchronization across all processes.
    #
    # Hint 1: Use torch.nn.parallel.DistributedDataParallel()
    # Hint 2: For CUDA, specify device_ids=[device.index] to pin to specific GPU
    # Hint 3: For CPU, omit device_ids parameter (set to None)
    # Hint 4: The wrapped model behaves like the original but syncs gradients automatically
    #
    # Key insight: This replaces your manual allreduce_grads() from Step 1!

    # TODO: Implement this section
    pass
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(ddp.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    return ddp, criterion, optim, device


def train_one_epoch(
    model, crit, optim, device, loader, *, hb_every: int = 50, is_master: bool = False
) -> Tuple[float, int]:
    model.train()
    loss_sum, n = 0.0, 0
    total = len(loader)
    for i, (xb, yb) in enumerate(loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        loss = crit(model(xb), yb)
        loss.backward()
        optim.step()
        bs = int(xb.size(0))
        loss_sum += float(loss.item()) * bs
        n += bs
        if is_master and (i == 1 or (hb_every > 0 and i % hb_every == 0) or i == total):
            print(
                f"\r[train] {i}/{total} loss={float(loss.item()):.4f}",
                end="",
                flush=True,
            )
    return loss_sum / max(1, n), n


def evaluate(model, crit, device, loader, world: int) -> Tuple[float, float, int]:
    import torch
    import torch.distributed as dist

    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss_sum += float(crit(logits, yb).item()) * int(xb.size(0))
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            n += int(xb.size(0))
    t = torch.tensor(
        [loss_sum, float(correct), float(n)], dtype=torch.float64, device="cpu"
    )
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    loss_sum, correct, n = float(t[0].item()), int(t[1].item()), int(t[2].item())
    return loss_sum / max(1, n), (correct / max(1, n)), n


def main(argv: list[str] | None = None) -> int:
    rank, world, backend = init_dist()
    dev = device_str()
    train_loader, val_loader, train_sampler, val_sampler = prepare_data(
        batch_size=128, rank=rank, world=world
    )
    if train_sampler is not None:
        try:
            train_sampler.set_epoch(1)
            if val_sampler is not None:
                val_sampler.set_epoch(1)
        except Exception:
            pass
    model, crit, optim, device = prepare_model_ddp(dev, world, rank)
    if rank == 0:
        print(
            f"Step 2 (teaching) • DDP • world={world} • backend={backend} • device={dev} • batches={len(train_loader)}",
            flush=True,
        )
    hb = int(os.getenv("HB_EVERY", "5"))
    train_loss, _ = train_one_epoch(
        model, crit, optim, device, train_loader, hb_every=hb, is_master=(rank == 0)
    )
    if rank == 0:
        print()  # newline after heartbeat
    val_loss, val_acc, _ = evaluate(model, crit, device, val_loader, world)
    if rank == 0:
        print(
            f"epoch 001 | ddp | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )

    # Tear down
    try:
        import torch.distributed as dist

        dist.destroy_process_group()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
