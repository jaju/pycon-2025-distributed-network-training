from __future__ import annotations

from typing import Tuple, Any
from dataclasses import dataclass

import os
import sys
import time

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


def setup_dist(backend: str = "gloo") -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


@dataclass(slots=True)
class RuntimeContext:
    rank: int
    world_size: int
    device: Any
    backend: str


def display_training_header(cfg: TrainingConfig, ctx: RuntimeContext) -> None:
    if ctx.rank != 0:
        return
    console.print(
        Rule(
            title=f"[bold cyan]Step 1 • Manual Distributed • {cfg.run.run_name}[/]",
            style="green",
        )
    )
    tbl = Table(show_header=False, box=None, pad_edge=False)
    tbl.add_column("Key", style="yellow", no_wrap=True)
    tbl.add_column("Value", style="white")
    tbl.add_row("device", f"[bold magenta]{cfg.run.device}[/]")
    tbl.add_row("backend", f"[bold red]{ctx.backend}[/]")
    tbl.add_row("epochs", f"[bold white]{cfg.optim.epochs}[/]")
    tbl.add_row("batch_size", f"[bold white]{cfg.data.batch_size}[/]")
    tbl.add_row("lr", f"[bold white]{cfg.optim.lr}[/]")
    tbl.add_row("scheduler", f"[bold cyan]{cfg.optim.scheduler}[/]")
    tbl.add_row("show_deltas", f"[bold white]{cfg.console.show_deltas}[/]")
    tbl.add_row("world_size", f"[bold white]{ctx.world_size}[/]")
    tbl.add_row("rank", f"[bold white]{ctx.rank}[/]")
    console.print(tbl)
    console.print(
        "[yellow]⚠️  Using manual gradient sync (naive, no bucketing/overlap) — compare with Step 0 and Step 2![/]"
    )


def prepare_monitor(cfg: TrainingConfig, ctx: RuntimeContext) -> MetricsMonitor:
    return MetricsMonitor(
        run_name=cfg.run.run_name,
        artifacts_dir=cfg.run.artifacts_dir,
        runs_dir=cfg.run.runs_dir,
        world_size=ctx.world_size,
        rank=ctx.rank,
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
        out = cfg.run.data_dir / f"step1_{cfg.run.run_name}_model.pt"
        torch.save(target.state_dict(), out)
    except Exception:
        pass


def _broadcast_model_buffers(model, src: int = 0) -> None:
    import torch
    import torch.distributed as dist

    backend = str(dist.get_backend()).lower()
    with torch.no_grad():
        for buf in model.buffers():
            if buf is None:
                continue
            orig_device = buf.device
            if backend == "nccl" and orig_device.type == "cuda":
                dist.broadcast(buf, src=src)
            else:
                tmp = buf.detach().to("cpu")
                dist.broadcast(tmp, src=src)
                buf.copy_(tmp.to(orig_device))


def _broadcast_model_params(model, src: int = 0) -> None:
    """Broadcast initial model parameters so all ranks start identically.

    For CUDA+NCCL we broadcast in-place on device. For other backends/devices,
    we round-trip through CPU to ensure the collective succeeds.
    """
    import torch
    import torch.distributed as dist

    backend = str(dist.get_backend()).lower()
    with torch.no_grad():
        for p in model.parameters():
            if p is None:
                continue
            orig_device = p.device
            if backend == "nccl" and orig_device.type == "cuda":
                dist.broadcast(p.data, src=src)
            else:
                tmp = p.data.detach().to("cpu")
                dist.broadcast(tmp, src=src)
                p.data.copy_(tmp.to(orig_device))


def prepare_data(cfg: TrainingConfig, rank: int, world_size: int):
    try:
        import torch.utils.data as data
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except Exception as e:  # noqa: BLE001
        if cfg.data.synthetic_if_missing:
            return _synthetic_loaders(cfg, rank, world_size)
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
    tx_val = transforms.Compose([transforms.ToTensor(), normalize])

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
            return _synthetic_loaders(cfg, rank, world_size)
        raise

    train_sampler = data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=cfg.run.seed,
        drop_last=False,
    )
    val_sampler = data.distributed.DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=(cfg.run.device == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=(cfg.run.device == "cuda"),
        drop_last=False,
        persistent_workers=False,
    )
    return train_loader, val_loader, train_sampler, val_sampler


def _synthetic_loaders(
    cfg: TrainingConfig, rank: int | None = None, world_size: int | None = None
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    x_train = torch.randn(cfg.data.synthetic_train_size, 3, 32, 32)
    y_train = torch.randint(0, 10, (cfg.data.synthetic_train_size,))
    x_val = torch.randn(cfg.data.synthetic_val_size, 3, 32, 32)
    y_val = torch.randint(0, 10, (cfg.data.synthetic_val_size,))

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    if (world_size or 1) > 1:
        assert rank is not None and world_size is not None
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.run.seed,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.data.batch_size, shuffle=False, sampler=val_sampler
    )
    return train_loader, val_loader, train_sampler, val_sampler


def prepare_model_and_optimizer(cfg: TrainingConfig):
    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device(cfg.run.device)
    model = models.resnet18(weights=None, num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
    )
    return model, criterion, optimizer, device


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


def manual_allreduce_gradients(model, world_size: int) -> tuple[float, int]:
    import torch.distributed as dist

    backend = str(dist.get_backend()).lower()
    if world_size <= 1:
        return 0.0, 0
    t0 = time.perf_counter()
    calls = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad
        device_type = g.device.type
        if (device_type == "cuda" and backend == "nccl") or (device_type == "cpu"):
            dist.all_reduce(g, op=dist.ReduceOp.SUM)
            g.div_(world_size)
            calls += 1
        else:
            tmp = g.detach().to("cpu")
            dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
            tmp.div_(world_size)
            g.copy_(tmp.to(g.device))
            calls += 1
    return (time.perf_counter() - t0), calls


def train_step_manual_sync(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    xb,
    yb,
    world_size: int,
):
    import time as _time

    t_iter0 = _time.perf_counter()
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()
    t_comm, calls = manual_allreduce_gradients(model, world_size)
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    bs = int(xb.size(0))
    it_ms = (_time.perf_counter() - t_iter0) * 1000.0
    return float(loss.item()), bs, float(it_ms), float(t_comm), int(calls)


def train_epoch_manual_sync(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    train_loader,
    world_size: int,
    *,
    progress,
    task_id: int,
    rank: int,
    worker_hb: bool,
    worker_hb_every: int,
):
    model.train()
    running_loss = 0.0
    n = 0
    comm_time = 0.0
    num_calls = 0
    for i, (xb, yb) in enumerate(train_loader, start=1):
        loss_item, bs, iter_dur_ms, comm_s, calls = train_step_manual_sync(
            model, criterion, optimizer, scheduler, device, xb, yb, world_size
        )
        running_loss += loss_item * bs
        n += bs
        comm_time += comm_s
        num_calls += calls
        progress.advance(task_id, 1)
        progress.update(
            task_id,
            description=f"step {i}: loss={loss_item:.4f} it={iter_dur_ms:.0f}ms",
        )
        if rank != 0 and worker_hb:
            if i == 1 or (i % max(1, worker_hb_every) == 0) or i == len(train_loader):
                try:
                    print(
                        f"[rank {rank}] step {i}: loss={loss_item:.4f} it={iter_dur_ms:.0f}ms",
                        flush=True,
                    )
                except Exception:
                    pass
    return running_loss / max(1, n), n, comm_time, num_calls


def evaluate(
    model, criterion, device, val_loader, world_size: int
) -> Tuple[float, int, int]:
    import torch
    import torch.distributed as dist

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
    if world_size > 1:
        metrics = torch.tensor(
            [loss_sum, float(correct), float(n)], device="cpu", dtype=torch.float64
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        loss_sum = float(metrics[0].item())
        correct = int(metrics[1].item())
        n = int(metrics[2].item())
    return loss_sum, correct, n


def run_training(
    cfg: TrainingConfig,
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    train_loader,
    val_loader,
    train_sampler,
    val_sampler,
    world_size: int,
    rank: int,
    monitor: MetricsMonitor,
) -> None:
    for epoch in range(1, cfg.optim.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)
        import time as _time

        _epoch_start = _time.perf_counter()
        monitor.epoch_start()
        with _make_progress(
            enabled=(rank == 0 and cfg.console.progress_bar)
        ) as progress:
            task = progress.add_task(f"epoch {epoch}", total=len(train_loader))
            train_loss, n_train, comm_s, calls = train_epoch_manual_sync(
                model,
                criterion,
                optimizer,
                scheduler,
                device,
                train_loader,
                world_size,
                progress=progress,
                task_id=task,
                rank=rank,
                worker_hb=bool(cfg.console.worker_heartbeat),
                worker_hb_every=int(cfg.console.worker_heartbeat_every),
            )

        if cfg.dist.broadcast_buffers and world_size > 1:
            _broadcast_model_buffers(model, src=0)

        loss_sum, correct, n_val = evaluate(
            model, criterion, device, val_loader, world_size
        )
        import torch
        import torch.distributed as dist

        local_elapsed = _time.perf_counter() - _epoch_start
        t_elapsed = torch.tensor([local_elapsed], dtype=torch.float64, device="cpu")
        t_samples = torch.tensor([float(n_train)], dtype=torch.float64, device="cpu")
        t_comm = torch.tensor([float(comm_s)], dtype=torch.float64, device="cpu")
        if world_size > 1:
            dist.all_reduce(t_elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(t_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_comm, op=dist.ReduceOp.MAX)
        global_elapsed = float(t_elapsed.item())
        global_samples = int(t_samples.item())
        global_comm = float(t_comm.item())
        global_throughput = (
            (global_samples / global_elapsed) if global_elapsed > 0 else None
        )
        if rank == 0:
            g_loss = float(loss_sum) / max(1.0, float(n_val))
            g_acc = float(correct) / max(1.0, float(n_val))
        else:
            g_loss = None
            g_acc = None

        current_lr = optimizer.param_groups[0]["lr"]
        monitor.epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=g_loss,
            val_acc=g_acc,
            num_samples=n_train,
            lr=float(current_lr),
            comm_overhead_s=comm_s,
            num_allreduces=calls,
            global_epoch_sec=global_elapsed if rank == 0 else None,
            global_samples=global_samples if rank == 0 else None,
            global_throughput=global_throughput if rank == 0 else None,
            comm_overhead_global_s=global_comm if rank == 0 else None,
        )


def main(argv: list[str] | None = None) -> int:
    import torch.distributed as dist

    cfg = TrainingConfig.from_argv(argv)
    cfg.apply_seeds()

    rank, world_size = setup_dist(backend=os.getenv("BACKEND", "gloo"))
    model, criterion, optimizer, device = prepare_model_and_optimizer(cfg)
    # Ensure identical initialization across ranks for manual sync training
    if world_size > 1:
        _broadcast_model_params(model, src=0)
    ctx = RuntimeContext(
        rank=rank, world_size=world_size, device=device, backend=str(dist.get_backend())
    )

    monitor = prepare_monitor(cfg, ctx)
    display_training_header(cfg, ctx)

    train_loader, val_loader, train_sampler, val_sampler = prepare_data(
        cfg, ctx.rank, ctx.world_size
    )
    scheduler = prepare_scheduler(cfg, optimizer, len(train_loader))

    run_training(
        cfg,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
        ctx.world_size,
        ctx.rank,
        monitor,
    )

    monitor.finalize()
    monitor.save_json()
    monitor.plot()
    _save_model("step1", cfg, model, rank=ctx.rank)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
