from __future__ import annotations

from typing import Tuple, Any
from dataclasses import dataclass

import os
import sys
from math import ceil

from rich.console import Console
from rich.rule import Rule
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.table import Table

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


def _param_grad_bytes(model) -> int:
    total = 0
    for p in model.parameters():
        if p.requires_grad:
            total += int(p.numel()) * int(p.element_size())
    return total


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


def setup_distributed_training(cfg: TrainingConfig) -> tuple[RuntimeContext, Any, Any]:
    import torch.distributed as dist

    if cfg.run.device == "mps":
        raise RuntimeError(
            "DDP on MPS is not supported with gloo. Use --device cpu for DDP on Apple Silicon or CUDA+NCCL."
        )
    backend = os.getenv("BACKEND") or ("nccl" if cfg.run.device == "cuda" else "gloo")
    rank, world_size = setup_dist(backend=backend)
    local_rank_env = os.getenv("LOCAL_RANK")
    local_rank = int(local_rank_env) if local_rank_env is not None else None
    model, criterion, device = prepare_base_model(cfg, local_rank)
    backend_str = str(dist.get_backend())
    ctx = RuntimeContext(
        rank=rank, world_size=world_size, device=device, backend=backend_str
    )
    return ctx, model, criterion


def create_ddp_model(
    model: Any, cfg: TrainingConfig, ctx: RuntimeContext
) -> tuple[Any, Any]:
    import torch.nn as nn
    import torch as _torch

    if int(ctx.world_size) <= 1:
        optimizer = _torch.optim.SGD(
            model.parameters(),
            lr=cfg.optim.lr,
            momentum=cfg.optim.momentum,
            weight_decay=cfg.optim.weight_decay,
        )
        return model, optimizer

    ddp_kwargs: dict = {
        "find_unused_parameters": bool(cfg.ddp.find_unused_parameters),
        "broadcast_buffers": True,
        "static_graph": bool(cfg.ddp.static_graph),
        "bucket_cap_mb": int(cfg.ddp.bucket_cap_mb)
        if int(cfg.ddp.bucket_cap_mb) > 0
        else 25,
        "gradient_as_bucket_view": (
            bool(getattr(cfg.ddp, "gradient_as_bucket_view", True))
            if getattr(ctx.device, "type", "cpu") == "cuda"
            else False
        ),
    }
    try:
        if ctx.device.type == "cuda":
            ddp_model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[ctx.device.index],
                output_device=ctx.device.index,
                **ddp_kwargs,
            )
        else:
            ddp_model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    except TypeError:
        fallback_kwargs = {
            k: v
            for k, v in ddp_kwargs.items()
            if k in {"find_unused_parameters", "broadcast_buffers", "bucket_cap_mb"}
        }
        if ctx.device.type == "cuda":
            ddp_model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[ctx.device.index],
                output_device=ctx.device.index,
                **fallback_kwargs,
            )
        else:
            ddp_model = nn.parallel.DistributedDataParallel(model, **fallback_kwargs)

    optimizer = _torch.optim.SGD(
        ddp_model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
    )
    return ddp_model, optimizer


def setup_monitoring(
    cfg: TrainingConfig, ctx: RuntimeContext, model: Any, ddp_model: Any
) -> tuple[MetricsMonitor, dict]:
    monitor = MetricsMonitor(
        run_name=cfg.run.run_name,
        artifacts_dir=cfg.run.artifacts_dir,
        runs_dir=cfg.run.runs_dir,
        world_size=ctx.world_size,
        rank=ctx.rank,
        show_deltas=cfg.console.show_deltas,
    )
    if bool(cfg.instr.ddp_comm_stats):
        try:
            monitor.attach_ddp(ddp_model)
        except Exception:
            pass
    grad_total_bytes = _param_grad_bytes(model)
    expected_buckets = int(
        ceil(grad_total_bytes / float(max(1, int(cfg.ddp.bucket_cap_mb)) * 1024 * 1024))
    )
    min_comm_per_iter = (
        (grad_total_bytes * (2.0 * (ctx.world_size - 1) / ctx.world_size))
        if ctx.world_size > 1
        else 0.0
    )
    proxies = {
        "grad_total_bytes": grad_total_bytes,
        "expected_buckets": expected_buckets,
        "min_comm_per_iter": min_comm_per_iter,
    }
    return monitor, proxies


def display_training_header(
    cfg: TrainingConfig, ctx: RuntimeContext, proxies: dict
) -> None:
    if ctx.rank != 0:
        return
    console.print(
        Rule(title=f"[bold cyan]Step 2 • DDP • {cfg.run.run_name}[/]", style="green")
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
    tbl.add_row("bucket_cap_mb", f"[bold white]{cfg.ddp.bucket_cap_mb}[/]")
    tbl.add_row(
        "grad_total_mb", f"[bold white]{proxies['grad_total_bytes'] / 1e6:.2f}[/]"
    )
    tbl.add_row("expected_buckets", f"[bold white]{proxies['expected_buckets']}[/]")
    console.print(tbl)
    console.print(
        "[yellow]✅ DDP enabled (bucketed, overlapped gradients). Compare with Step 1.[/]"
    )


def _save_model(
    step_prefix: str, cfg: TrainingConfig, model, rank: int | None = None
) -> None:
    if rank is not None and int(rank) != 0:
        return
    try:
        import torch

        target = model.module if hasattr(model, "module") else model
        out = cfg.run.data_dir / f"step2_{cfg.run.run_name}_model.pt"
        torch.save(target.state_dict(), out)
    except Exception:
        pass


def _profiler_start(cfg: TrainingConfig, ctx: RuntimeContext):
    if not bool(cfg.instr.profiler_on):
        return None
    try:
        import torch.profiler as profiler  # type: ignore

        p = profiler.profile(
            schedule=profiler.schedule(
                wait=0,
                warmup=max(0, int(cfg.instr.profiler_warmup_steps)),
                active=max(1, int(cfg.instr.profiler_active_steps)),
            ),
            on_trace_ready=None,
            record_shapes=False,
            with_stack=False,
            profile_memory=False,
            with_modules=False,
        )
        p.__enter__()
        return p
    except Exception:
        return None


def _profiler_step(prof) -> None:
    if prof is None:
        return
    try:
        prof.step()
    except Exception:
        pass


def _profiler_close_and_aggregate(prof) -> tuple[float, int]:
    if prof is None:
        return 0.0, 0
    comm_time_s = 0.0
    num_calls = 0
    try:
        prof.__exit__(None, None, None)
        try:
            events = prof.key_averages()
            for ev in events:
                name = str(getattr(ev, "key", getattr(ev, "name", ""))).lower()
                if any(k in name for k in ("allreduce", "all_reduce", "c10d", "nccl")):
                    comm_time_s += float(getattr(ev, "self_cpu_time_total", 0.0)) / 1e6
                    num_calls += int(getattr(ev, "count", 0))
        except Exception:
            pass
    except Exception:
        pass
    return comm_time_s, num_calls


def compute_global_epoch_aggregates(
    local_elapsed: float,
    n_train: int,
    ctx: RuntimeContext,
    ddp_comm_overhead_s: float | None,
):
    import torch
    import torch.distributed as dist

    t_elapsed = torch.tensor([local_elapsed], dtype=torch.float64, device="cpu")
    t_samples = torch.tensor([float(n_train)], dtype=torch.float64, device="cpu")
    t_comm = torch.tensor(
        [float(ddp_comm_overhead_s or 0.0)], dtype=torch.float64, device="cpu"
    )
    if ctx.world_size > 1:
        dist.all_reduce(t_elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(t_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(t_comm, op=dist.ReduceOp.MAX)
    global_elapsed = float(t_elapsed.item())
    global_samples = int(t_samples.item())
    global_comm = float(t_comm.item()) if (ddp_comm_overhead_s is not None) else None
    global_throughput = (
        (global_samples / global_elapsed) if global_elapsed > 0 else None
    )
    return global_elapsed, global_samples, global_throughput, global_comm


def compute_speedup_efficiency(
    cfg: TrainingConfig,
    ctx: RuntimeContext,
    global_elapsed: float | None,
    global_throughput: float | None,
):
    speedup = None
    efficiency = None
    if ctx.rank == 0 and global_elapsed is not None:
        if cfg.instr.baseline_epoch_sec and global_elapsed > 0:
            speedup = float(cfg.instr.baseline_epoch_sec) / float(global_elapsed)
        elif cfg.instr.baseline_throughput and global_throughput:
            speedup = float(global_throughput) / float(cfg.instr.baseline_throughput)
        if speedup is not None and ctx.world_size > 0:
            efficiency = speedup / ctx.world_size
    return speedup, efficiency


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


def prepare_base_model(cfg: TrainingConfig, local_rank: int | None):
    import os
    import torch
    import torch.nn as nn
    from torchvision import models

    if cfg.run.device == "cuda" and torch.cuda.is_available():
        idx = (
            int(local_rank)
            if local_rank is not None
            else int(os.getenv("LOCAL_RANK", "0"))
        )
        torch.cuda.set_device(idx)
        device = torch.device("cuda", idx)
    elif (
        cfg.run.device == "mps"
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = models.resnet18(weights=None, num_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    return model, criterion, device


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


def _train_step_ddp(
    model, criterion, optimizer, scheduler, device, xb, yb, *, timers
) -> tuple[float, int, float]:
    import time as _time

    t_iter0 = _time.perf_counter()
    timers.iter_start()
    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with timers.forward():
        logits = model(xb)
        loss = criterion(logits, yb)
    with timers.backward():
        loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    timers.iter_end()
    bs = int(xb.size(0))
    it_ms = (_time.perf_counter() - t_iter0) * 1000.0
    return float(loss.item()), bs, float(it_ms)


def train_epoch_ddp(
    ddp_model,
    criterion,
    optimizer,
    scheduler,
    device,
    train_loader,
    *,
    timers,
    progress,
    task_id: int,
    rank: int,
    worker_hb: bool,
    worker_hb_every: int,
):
    running_loss = 0.0
    n = 0
    forward_s = 0.0
    backward_s = 0.0
    opt_s = 0.0
    iter_ms: list[float] = []
    for i, (xb, yb) in enumerate(train_loader, start=1):
        loss_item, bs, iter_dur_ms = _train_step_ddp(
            ddp_model, criterion, optimizer, scheduler, device, xb, yb, timers=timers
        )
        forward_s = float(getattr(timers, "forward_sec", forward_s))
        backward_s = float(getattr(timers, "backward_sec", backward_s))
        running_loss += loss_item * bs
        n += bs
        iter_ms.append(iter_dur_ms)
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
    t_epoch_train = forward_s + backward_s + opt_s
    return (
        running_loss / max(1, n),
        n,
        iter_ms,
        forward_s,
        backward_s,
        opt_s,
        t_epoch_train,
    )


def evaluate(
    model, criterion, val_loader, ctx: RuntimeContext
) -> Tuple[float, int, int]:
    import torch
    import torch.distributed as dist

    model.eval()
    loss_sum = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(ctx.device, non_blocking=True)
            yb = yb.to(ctx.device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            bs = xb.size(0)
            loss_sum += loss.item() * bs
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            n += bs
    if ctx.world_size > 1:
        backend = ctx.backend.lower()
        metrics = torch.tensor(
            [loss_sum, float(correct), float(n)],
            device=(
                ctx.device if ctx.device.type == "cuda" and backend == "nccl" else "cpu"
            ),
            dtype=torch.float64,
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        loss_sum = float(metrics[0].item())
        correct = int(metrics[1].item())
        n = int(metrics[2].item())
    return loss_sum, correct, n


def main(argv: list[str] | None = None) -> int:
    cfg = TrainingConfig.from_argv(argv)
    cfg.apply_seeds()
    ctx, model, criterion = setup_distributed_training(cfg)
    ddp_model, optimizer = create_ddp_model(model, cfg, ctx)
    train_loader, val_loader, train_sampler, val_sampler = prepare_data(
        cfg, ctx.rank, ctx.world_size
    )
    scheduler = prepare_scheduler(cfg, optimizer, len(train_loader))
    monitor, proxies = setup_monitoring(cfg, ctx, model, ddp_model)
    display_training_header(cfg, ctx, proxies)
    timers = monitor.attach_timers(ctx.device, enabled=bool(cfg.instr.measure_phases))

    import time as _time

    prof = _profiler_start(cfg, ctx)
    prof_comm_time = 0.0
    prof_comm_calls = 0

    for epoch in range(1, cfg.optim.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        _epoch_start = _time.perf_counter()
        monitor.epoch_start()
        with _make_progress(
            enabled=(ctx.rank == 0 and cfg.console.progress_bar)
        ) as progress:
            task = progress.add_task(f"epoch {epoch}", total=len(train_loader))
            train_loss, n_train, iter_ms, f_s, b_s, o_s, t_train = train_epoch_ddp(
                ddp_model,
                criterion,
                optimizer,
                scheduler,
                ctx.device,
                train_loader,
                timers=timers,
                progress=progress,
                task_id=task,
                rank=ctx.rank,
                worker_hb=bool(cfg.console.worker_heartbeat),
                worker_hb_every=int(cfg.console.worker_heartbeat_every),
            )

        _profiler_step(prof)

        loss_sum, correct, n_val = evaluate(ddp_model, criterion, val_loader, ctx)

        local_elapsed = _time.perf_counter() - _epoch_start
        ddp_comm_overhead_s = monitor.local_ddp_comm_overhead_s()
        global_elapsed, global_samples, global_throughput, global_comm = (
            compute_global_epoch_aggregates(
                local_elapsed, n_train, ctx, ddp_comm_overhead_s
            )
        )

        if ctx.rank == 0:
            g_loss = float(loss_sum) / max(1.0, float(n_val))
            g_acc = float(correct) / max(1.0, float(n_val))
        else:
            g_loss = None
            g_acc = None

        speedup, efficiency = compute_speedup_efficiency(
            cfg, ctx, global_elapsed if ctx.rank == 0 else None, global_throughput
        )

        current_lr = optimizer.param_groups[0]["lr"]
        monitor.epoch_end(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=g_loss,
            val_acc=g_acc,
            num_samples=n_train,
            lr=float(current_lr),
            forward_sec=f_s,
            backward_sec=b_s,
            optimizer_sec=o_s,
            total_train_sec=t_train,
            iter_ms=iter_ms,
            ddp_num_buckets=None,
            ddp_avg_bucket_size_mb=None,
            ddp_bucket_time_ms=None,
            ddp_comm_overhead_s=ddp_comm_overhead_s,
            ddp_expected_bucket_count=proxies["expected_buckets"],
            ddp_grad_total_bytes=proxies["grad_total_bytes"],
            ddp_min_comm_volume_bytes_per_iter=proxies["min_comm_per_iter"],
            global_epoch_sec=global_elapsed if ctx.rank == 0 else None,
            global_samples=global_samples if ctx.rank == 0 else None,
            global_throughput=global_throughput if ctx.rank == 0 else None,
            comm_overhead_global_s=global_comm if ctx.rank == 0 else None,
            speedup=speedup if ctx.rank == 0 else None,
            efficiency=efficiency if ctx.rank == 0 else None,
        )

    comm_s, calls = _profiler_close_and_aggregate(prof)
    if calls or comm_s:
        prof_comm_time += comm_s
        prof_comm_calls += calls
        monitor.log_profiler_metrics(prof_comm_time, prof_comm_calls)

    monitor.finalize()
    monitor.save_json()
    monitor.plot()
    _save_model("step2", cfg, ddp_model, rank=ctx.rank)

    import torch.distributed as dist

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
