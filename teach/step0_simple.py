"""
Step 0 (Teaching): Minimal single-process training loop.

- ResNet18 on CIFAR-10 if available in DATA_DIR (no auto-download)
- Falls back to a small synthetic dataset if CIFAR-10 is unavailable
- One concise file: data → model → train → eval → print metrics
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

from teach.common import device_str, load_dataset


def prepare_data(batch_size: int = 128):
    from torch.utils.data import DataLoader

    data_dir = os.getenv("DATA_DIR", "data")
    train_ds, val_ds = load_dataset(data_dir)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def prepare_model(device_str: str):
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device(device_str)
    model = models.resnet18(weights=None, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    return model, criterion, optimizer, device


def train_one_epoch(
    model, criterion, optimizer, device, train_loader, *, hb_every: int = 50
) -> Tuple[float, int]:
    model.train()
    loss_sum, n = 0.0, 0
    total = len(train_loader)
    for i, (xb, yb) in enumerate(train_loader, start=1):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
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


def evaluate(model, criterion, device, val_loader) -> Tuple[float, float, int]:
    import torch

    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = int(xb.size(0))
            loss_sum += float(loss.item()) * bs
            correct += int((logits.argmax(dim=1) == yb).sum().item())
            n += bs
    return loss_sum / max(1, n), (correct / max(1, n)), n


def main(argv: list[str] | None = None) -> int:
    device = device_str()
    train_loader, val_loader = prepare_data(batch_size=128)
    model, criterion, optimizer, device_obj = prepare_model(device)
    print(
        f"Step 0 (teaching) • device={device} • batches={len(train_loader)}", flush=True
    )
    hb_every = int(os.getenv("HB_EVERY", "5"))
    train_loss, n_train = train_one_epoch(
        model, criterion, optimizer, device_obj, train_loader, hb_every=hb_every
    )
    print()  # newline after heartbeat
    val_loss, val_acc, _ = evaluate(model, criterion, device_obj, val_loader)

    # Rank-0 style print
    print(
        f"epoch 001 | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc * 100:.2f}% | device={device}"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
