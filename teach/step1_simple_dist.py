"""Step 1 (Teaching): Minimal manual data-parallel with heartbeats.

- CIFAR-10 if available in DATA_DIR (no auto-download); synthetic fallback otherwise
- Optional distributed if RANK/WORLD_SIZE are set (gloo)
- Rank 0 prints heartbeats and a one-line summary
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch.distributed as dist

from teach.common import device_str, init_dist, load_dataset


def prepare_data(rank: int, world_size: int, batch_size: int = 128):
    from torch.utils.data import DataLoader, Subset

    data_dir = os.getenv("DATA_DIR", "data")
    train_ds, val_ds = load_dataset(data_dir)

    # Simple manual shard for both training and validation sets
    if world_size > 1:
        # Given the world size, split both datasets into equal shards
        # and assign one shard to this rank.
        # 1. Split the training and validation datasets into `world_size` shards
        # 2. Assign shard `rank` to this process
        # Simple arithmetic, and use torch.utils.data.Subset to create the shard.
        n_train = len(train_ds)
        shard = n_train // world_size
        start, end = rank * shard, rank * shard + shard
        train_ds = Subset(train_ds, list(range(start, end)))

        # Shard validation dataset similarly
        # TODO: Implement this section

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def prepare_model(dev: str, world_size: int, rank: int):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torchvision import models

    device = torch.device(dev)
    model = models.resnet18(weights=None, num_classes=10).to(device)

    # Synchronize model weights across all ranks
    if world_size > 1:
        if rank == 0:
            print("Broadcasting initial weights from rank 0...", flush=True)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        if rank == 0:
            print("Weight broadcast complete.", flush=True)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    return model, criterion, optim, device


def allreduce_grads(model, world_size: int) -> None:
    if world_size <= 1:
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)


def train_one_epoch(
    model,
    criterion,
    optim,
    device,
    train_loader,
    rank: int,
    world: int,
    *,
    hb_every: int = 50,
) -> Tuple[float, int]:
    model.train()
    loss_sum, n = 0.0, 0
    total = len(train_loader)
    for i, (xb, yb) in enumerate(train_loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        allreduce_grads(model, world)
        optim.step()
        bs = int(xb.size(0))
        loss_sum += float(loss.item()) * bs
        n += bs
        if rank == 0 and (i == 1 or (hb_every > 0 and i % hb_every == 0) or i == total):
            print(
                f"\r[train] {i}/{total} loss={float(loss.item()):.4f}",
                end="",
                flush=True,
            )
    return loss_sum / max(1, n), n


def evaluate(
    model, criterion, device, val_loader, rank: int, world: int
) -> Tuple[float, float, int]:
    import torch
    import torch.distributed as dist

    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            count += int(xb.size(0))
            logits = model(xb)
            loss_sum += float(criterion(logits, yb).item()) * int(xb.size(0))
            correct += int((logits.argmax(dim=1) == yb).sum().item())
    if world > 1:
        # DISTRIBUTED EVALUATION CHALLENGE:
        #
        # Problem: Each rank has computed local metrics (loss_sum, correct, n) on its
        # shard of the validation set. But we need GLOBAL metrics across all data.
        #
        # Without aggregation: Each rank would report different accuracy/loss values!
        # This breaks the fundamental assumption that validation metrics reflect
        # performance on the complete dataset.
        #
        # Your task: Aggregate the three local values across all ranks to get global totals.
        #
        # Hint 1: Use torch.tensor() to pack the three values into a single tensor
        # Hint 2: Use torch.distributed.all_reduce() with SUM operation to add up across ranks
        # Hint 3: For all_reduce to work, tensor must be on CPU and use float64 dtype
        # Hint 4: Unpack the aggregated tensor back into individual variables
        #
        # Pattern: [local_val1, local_val2, local_val3] -> all_reduce(SUM) -> [global_val1, global_val2, global_val3]

        t = torch.tensor(
            [loss_sum, float(correct), float(count)], dtype=torch.float64, device="cpu"
        )
        # TODO: Implement this section
        pass
    return loss_sum / max(1, count), (correct / max(1, count)), count


def main(argv: list[str] | None = None) -> int:
    rank, world_size, inited = init_dist()
    dev = device_str()
    train_loader, val_loader = prepare_data(rank, world_size, batch_size=128)
    model, crit, optim, device = prepare_model(dev, world_size, rank)
    print(
        f"Step 1 (teaching) • world={world_size} • device={dev} • batches={len(train_loader)}",
        flush=True,
    )
    hb = int(os.getenv("HB_EVERY", "5"))
    train_loss, _ = train_one_epoch(
        model, crit, optim, device, train_loader, rank, world_size, hb_every=hb
    )
    val_loss, val_acc, _ = evaluate(model, crit, device, val_loader, rank, world_size)
    if rank == 0:
        print()
        print(
            f"epoch 001 | mode={'manual-sync' if world_size > 1 else 'single'} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
        )
    if inited:
        try:
            import torch.distributed as dist

            dist.destroy_process_group()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
