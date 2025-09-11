from __future__ import annotations

from typing import Tuple

import sys

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)

from reference.config import TrainingConfig
from reference.metrics import MetricsMonitor


console = Console(width=120, force_terminal=True, color_system="standard")


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, *args, **kwargs):
        return 0

    def advance(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None


def _make_progress(enabled: bool):
    if not enabled:
        return _NullProgress()
    return Progress(
        TextColumn("{task.description}", justify="left"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=True,
    )


def prepare_data(cfg: TrainingConfig):
    try:
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except Exception as e:  # noqa: BLE001
        if cfg.data.synthetic_if_missing:
            return _synthetic_loaders(cfg)
        raise RuntimeError(
            "Torch/torchvision not available and synthetic fallback disabled"
        ) from e

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    tx_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    tx_val = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    try:
        train_ds = datasets.CIFAR10(
            root=str(cfg.run.data_dir),
            train=True,
            download=cfg.data.download_data,
            transform=tx_train,
        )
        val_ds = datasets.CIFAR10(
            root=str(cfg.run.data_dir),
            train=False,
            download=cfg.data.download_data,
            transform=tx_val,
        )
    except Exception:
        if cfg.data.synthetic_if_missing:
            return _synthetic_loaders(cfg)
        raise

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(cfg.run.device == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(cfg.run.device == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )
    return train_loader, val_loader


def _synthetic_loaders(cfg: TrainingConfig):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    x_train = torch.randn(cfg.data.synthetic_train_size, 3, 32, 32)
    y_train = torch.randint(0, 10, (cfg.data.synthetic_train_size,))
    x_val = torch.randn(cfg.data.synthetic_val_size, 3, 32, 32)
    y_val = torch.randint(0, 10, (cfg.data.synthetic_val_size,))

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.data.batch_size)
    return train_loader, val_loader


def prepare_model(cfg: TrainingConfig):
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device(cfg.run.device)
    model = models.resnet18(weights=None, num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion, device


def prepare_optimizer(cfg: TrainingConfig, model):
    import torch

    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
    )


def prepare_scheduler(cfg: TrainingConfig, optimizer, steps_per_epoch: int):
    import torch

    name = (cfg.optim.scheduler or "none").lower()
    if name == "none" or steps_per_epoch <= 0:
        return None

    total_steps = max(1, cfg.optim.epochs * steps_per_epoch)
    if name == "cosine":
        t_max = cfg.optim.cosine_tmax if cfg.optim.cosine_tmax > 0 else total_steps
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=cfg.optim.cosine_eta_min
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, cfg.optim.step_size * steps_per_epoch),
            gamma=cfg.optim.gamma,
        )
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.optim.onecycle_max_lr
            if cfg.optim.onecycle_max_lr > 0
            else cfg.optim.lr,
            total_steps=total_steps,
            pct_start=cfg.optim.onecycle_pct_start,
            anneal_strategy="cos",
        )
    raise ValueError(f"Unknown scheduler: {cfg.optim.scheduler}")


def train_step_basic(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    xb,
    yb,
):
    import time as _time

    t_iter0 = _time.perf_counter()
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    bs = int(xb.size(0))
    it_ms = (_time.perf_counter() - t_iter0) * 1000.0
    return float(loss.item()), bs, float(it_ms)


def train_epoch_basic(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    train_loader,
    *,
    progress,
    task_id: int,
) -> Tuple[float, int]:
    model.train()
    running_loss = 0.0
    n = 0
    for i, (xb, yb) in enumerate(train_loader, start=1):
        loss_item, bs, it_ms = train_step_basic(
            model, criterion, optimizer, scheduler, device, xb, yb
        )
        running_loss += loss_item * bs
        n += bs
        progress.advance(task_id, 1)
        progress.update(
            task_id, description=f"step {i}: loss={loss_item:.4f} it={it_ms:.0f}ms"
        )
    return running_loss / max(1, n), n


def evaluate(model, criterion, device, val_loader) -> Tuple[float, float, int]:
    import torch

    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            loss_sum += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n += bs
    val_loss = loss_sum / max(1, n)
    val_acc = correct / max(1, n)
    return val_loss, val_acc, n


def run_training(
    cfg: TrainingConfig,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    train_loader,
    val_loader,
    monitor: MetricsMonitor,
) -> None:
    for epoch in range(1, cfg.optim.epochs + 1):
        monitor.epoch_start()
        with _make_progress(cfg.console.progress_bar) as progress:
            task = progress.add_task(f"epoch {epoch}", total=len(train_loader))
            train_loss, n_train = train_epoch_basic(
                model,
                criterion,
                optimizer,
                scheduler,
                device,
                train_loader,
                progress=progress,
                task_id=task,
            )
        val_loss, val_acc, _ = evaluate(model, criterion, device, val_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        monitor.epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            num_samples=n_train,
            lr=float(current_lr),
        )


def display_training_header(cfg: TrainingConfig) -> None:
    console.print(
        Rule(
            title=f"[bold cyan]Step 0 • Baseline • {cfg.run.run_name}[/]", style="green"
        )
    )
    tbl = Table(show_header=False, box=None, pad_edge=False)
    tbl.add_column("Key", style="yellow", no_wrap=True)
    tbl.add_column("Value", style="white")
    tbl.add_row("device", f"[bold magenta]{cfg.run.device}[/]")
    tbl.add_row("epochs", f"[bold white]{cfg.optim.epochs}[/]")
    tbl.add_row("batch_size", f"[bold white]{cfg.data.batch_size}[/]")
    tbl.add_row("lr", f"[bold white]{cfg.optim.lr}[/]")
    tbl.add_row("scheduler", f"[bold cyan]{cfg.optim.scheduler}[/]")
    tbl.add_row("show_deltas", f"[bold white]{cfg.console.show_deltas}[/]")
    console.print(tbl)


def prepare_monitor(cfg: TrainingConfig) -> MetricsMonitor:
    return MetricsMonitor(
        run_name=cfg.run.run_name,
        artifacts_dir=cfg.run.artifacts_dir,
        runs_dir=cfg.run.runs_dir,
        show_deltas=cfg.console.show_deltas,
    )


def _save_model(
    step_prefix: str, cfg: TrainingConfig, model, rank: int | None = None
) -> None:
    if rank is not None and int(rank) != 0:
        return
    try:
        import torch

        target = model.module if hasattr(model, "module") else model
        out = cfg.run.data_dir / f"step0_{cfg.run.run_name}_model.pt"
        torch.save(target.state_dict(), out)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    cfg = TrainingConfig.from_argv(argv)
    cfg.apply_seeds()

    display_training_header(cfg)
    monitor = prepare_monitor(cfg)
    train_loader, val_loader = prepare_data(cfg)
    model, criterion, device = prepare_model(cfg)
    optimizer = prepare_optimizer(cfg, model)
    steps_per_epoch = len(train_loader)
    scheduler = prepare_scheduler(cfg, optimizer, steps_per_epoch)

    run_training(
        cfg,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        train_loader,
        val_loader,
        monitor,
    )

    monitor.finalize()
    monitor.save_json()
    monitor.plot()
    _save_model("step0", cfg, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
