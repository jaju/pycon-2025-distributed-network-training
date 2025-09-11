"""
Step 3 (Teaching): Minimal FSDP simulate (single process).

- No real sharding/collectives; prints proxies to explain FSDP concepts
- Uses CIFAR-10 from DATA_DIR if available (no auto-download); synthetic fallback otherwise
"""

from __future__ import annotations

import os
import sys
from typing import Tuple


def device_str() -> str:
    try:
        import torch

        d = os.getenv("DEVICE")
        if d:
            return d
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def prepare_data(batch_size: int = 128):
    from torch.utils.data import DataLoader
    from teach.common import datasets_for_teach

    data_dir = os.getenv("DATA_DIR", "data")
    train_ds, val_ds = datasets_for_teach(data_dir)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def prepare_model(dev: str):
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device(dev)
    model = models.resnet18(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    return model, criterion, optim, device


def param_bytes(model) -> int:
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += int(p.numel()) * int(p.element_size())
    return total


def train_one_epoch(
    model, crit, optim, device, loader, *, hb_every: int = 50
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
        if i == 1 or (hb_every > 0 and i % hb_every == 0) or i == total:
            print(
                f"\r[train] {i}/{total} loss={float(loss.item()):.4f}",
                end="",
                flush=True,
            )
    return loss_sum / max(1, n), n


def evaluate(model, crit, device, loader) -> Tuple[float, float, int]:
    import torch

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
    return loss_sum / max(1, n), (correct / max(1, n)), n


def main(argv: list[str] | None = None) -> int:
    dev = device_str()
    train_loader, val_loader = prepare_data(batch_size=128)
    model, crit, optim, device = prepare_model(dev)

    # FSDP proxies (simulate, world_size=1)
    total = float(param_bytes(model))
    shard_per_rank = total  # world_size=1
    expected_comm = 0.0  # 2*(w-1)/w * total, with w=1 → 0
    print(
        f"Step 3 (teaching) • FSDP (simulate) • device={dev} • param_total_mb={total / 1e6:.2f} • shard_mb_per_rank={shard_per_rank / 1e6:.2f} • expected_comm_mb_per_iter={expected_comm / 1e6:.2f}",
        flush=True,
    )

    hb = int(os.getenv("TEACH_HB_EVERY", "50"))
    train_loss, _ = train_one_epoch(
        model, crit, optim, device, train_loader, hb_every=hb
    )
    print()
    val_loss, val_acc, _ = evaluate(model, crit, device, val_loader)
    print(
        f"epoch 001 | fsdp(sim) | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}%"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
