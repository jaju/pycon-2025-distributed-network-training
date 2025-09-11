from __future__ import annotations

from typing import Any, Tuple
from dataclasses import dataclass
import os
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
        TextColumn("{task.description}"),
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
    simulate: bool


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

    if world_size > 1:
        import torch.utils.data as data

        train_sampler = data.distributed.DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=cfg.run.seed,
        )
        val_sampler = data.distributed.DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=(train_sampler is None),
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


def setup_distributed_training(cfg: TrainingConfig) -> tuple[RuntimeContext, Any, Any]:
    import torch.distributed as dist

    simulate = bool(cfg.fsdp.simulate) or (cfg.run.device != "cuda")
    if not simulate and cfg.run.device != "cuda":
        raise RuntimeError("FSDP requires CUDA. Set --fsdp-simulate for CPU pedagogy.")
    backend = os.getenv("BACKEND", "gloo" if simulate else "nccl")
    rank, world_size = setup_dist(backend=backend)
    if simulate and world_size > 1:
        raise RuntimeError("Simulate mode supports single-process only.")
    local_rank_env = os.getenv("LOCAL_RANK")
    local_rank = int(local_rank_env) if local_rank_env is not None else None
    model, criterion, device = prepare_base_model(cfg, local_rank)
    ctx = RuntimeContext(
        rank=rank,
        world_size=world_size,
        device=device,
        backend=str(dist.get_backend()),
        simulate=simulate,
    )
    return ctx, model, criterion


def _fsdp_sharding(cfg: TrainingConfig):
    from torch.distributed.fsdp import ShardingStrategy

    strat_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    return strat_map.get(
        cfg.fsdp.sharding_strategy.upper(), ShardingStrategy.FULL_SHARD
    )


def _fsdp_auto_wrap(cfg: TrainingConfig):
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

    if int(cfg.fsdp.auto_wrap_threshold) <= 0:
        return None
    return size_based_auto_wrap_policy(min_num_params=int(cfg.fsdp.auto_wrap_threshold))


def _fsdp_backward_prefetch(cfg: TrainingConfig):
    from torch.distributed.fsdp import BackwardPrefetch

    return (
        BackwardPrefetch.BACKWARD_PRE
        if cfg.fsdp.backward_prefetch.upper() == "BACKWARD_PRE"
        else BackwardPrefetch.BACKWARD_POST
    )


def _fsdp_mixed_precision_policy(cfg: TrainingConfig):
    try:
        from torch.distributed.fsdp import MixedPrecision  # type: ignore
        import torch as _torch

        mp_cfg = (cfg.fsdp.mixed_precision or "none").lower()
        if mp_cfg in ("fp16", "bf16"):
            dtype = _torch.float16 if mp_cfg == "fp16" else _torch.bfloat16
            return MixedPrecision(
                param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
            )
    except Exception:
        return None
    return None


def _fsdp_cpu_offload(cfg: TrainingConfig):
    try:
        from torch.distributed.fsdp import CPUOffload  # type: ignore

        if bool(cfg.fsdp.cpu_offload):
            return CPUOffload(offload_params=True)
    except Exception:
        return None
    return None


def _wrap_with_fsdp(model: Any, cfg: TrainingConfig, ctx: RuntimeContext):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    fsdp = FSDP(
        model,
        device_id=(
            ctx.device.index if getattr(ctx.device, "type", "cpu") == "cuda" else None
        ),
        sharding_strategy=_fsdp_sharding(cfg),
        auto_wrap_policy=_fsdp_auto_wrap(cfg),
        mixed_precision=_fsdp_mixed_precision_policy(cfg),
        cpu_offload=_fsdp_cpu_offload(cfg),
        use_orig_params=bool(cfg.fsdp.use_orig_params),
        forward_prefetch=bool(cfg.fsdp.forward_prefetch),
        backward_prefetch=_fsdp_backward_prefetch(cfg),
        limit_all_gathers=bool(cfg.fsdp.limit_all_gathers),
    )
    return fsdp


def create_fsdp_model(model: Any, cfg: TrainingConfig, ctx: RuntimeContext):
    if ctx.simulate:
        from torch.optim import SGD

        return (
            model,
            SGD(
                model.parameters(),
                lr=cfg.optim.lr,
                momentum=cfg.optim.momentum,
                weight_decay=cfg.optim.weight_decay,
            ),
            0,
        )
    fsdp_model = _wrap_with_fsdp(model, cfg, ctx)
    from torch.optim import SGD

    optimizer = SGD(
        fsdp_model.parameters(),
        lr=cfg.optim.lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
    )
    wrapped_count = sum(
        1 for m in fsdp_model.modules() if m.__class__ is fsdp_model.__class__
    )
    return fsdp_model, optimizer, wrapped_count


def setup_monitoring(
    cfg: TrainingConfig,
    ctx: RuntimeContext,
    model: Any,
    fsdp_model: Any,
    wrapped_count: int,
) -> tuple[MetricsMonitor, dict]:
    monitor = MetricsMonitor(
        run_name=cfg.run.run_name,
        artifacts_dir=cfg.run.artifacts_dir,
        runs_dir=cfg.run.runs_dir,
        world_size=ctx.world_size,
        rank=ctx.rank,
        show_deltas=cfg.console.show_deltas,
    )

    def _bytes_of_params(m) -> int:
        total = 0
        for p in m.parameters():
            if p.requires_grad:
                total += int(p.numel()) * int(p.element_size())
        return total

    param_total = _bytes_of_params(model)
    shard_bytes = (
        (param_total / ctx.world_size) if ctx.world_size > 0 else float(param_total)
    )
    opt_state_bytes = (
        int(param_total) if cfg.optim.momentum and cfg.optim.momentum > 0 else 0
    )
    expected_comm = (
        float(param_total) * (2.0 * (ctx.world_size - 1) / ctx.world_size)
        if ctx.world_size > 1
        else 0.0
    )
    proxies = {
        "fsdp_param_total_bytes": int(param_total),
        "fsdp_shard_bytes_per_rank": float(shard_bytes),
        "fsdp_opt_state_bytes": int(opt_state_bytes),
        "fsdp_expected_comm_per_iter_bytes": float(expected_comm),
        "fsdp_wrapped_modules": wrapped_count if wrapped_count > 0 else None,
        "fsdp_sharding_strategy": cfg.fsdp.sharding_strategy,
    }
    return monitor, proxies


def display_training_header(
    cfg: TrainingConfig, ctx: RuntimeContext, proxies: dict
) -> None:
    if ctx.rank != 0:
        return
    title = "Step 3 • FSDP (simulate)" if ctx.simulate else "Step 3 • FSDP"
    console.print(
        Rule(title=f"[bold cyan]{title} • {cfg.run.run_name}[/]", style="green")
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
    tbl.add_row(
        "fsdp/sharding_strategy", f"[bold white]{cfg.fsdp.sharding_strategy}[/]"
    )
    tbl.add_row(
        "fsdp/auto_wrap_threshold", f"[bold white]{cfg.fsdp.auto_wrap_threshold}[/]"
    )
    tbl.add_row("fsdp/use_orig_params", f"[bold white]{cfg.fsdp.use_orig_params}[/]")
    tbl.add_row("fsdp/forward_prefetch", f"[bold white]{cfg.fsdp.forward_prefetch}[/]")
    tbl.add_row(
        "fsdp/backward_prefetch", f"[bold white]{cfg.fsdp.backward_prefetch}[/]"
    )
    tbl.add_row(
        "fsdp/param_total_mb",
        f"[bold white]{proxies['fsdp_param_total_bytes'] / 1e6:.2f}[/]",
    )
    tbl.add_row(
        "fsdp/shard_mb_per_rank",
        f"[bold white]{proxies['fsdp_shard_bytes_per_rank'] / 1e6:.2f}[/]",
    )
    tbl.add_row(
        "fsdp/expected_comm_mb_per_iter",
        f"[bold white]{proxies['fsdp_expected_comm_per_iter_bytes'] / 1e6:.2f}[/]",
    )
    console.print(tbl)
    note = (
        "[yellow]ℹ️  Simulate mode: proxies only; run on CUDA+NCCL for real FSDP comm metrics.[/]"
        if ctx.simulate
        else "[yellow]✅ FSDP enabled (parameter sharding). Compare with Step 2![/]"
    )
    console.print(note)


def _save_model(
    step_prefix: str, cfg: TrainingConfig, model, rank: int | None = None
) -> None:
    if rank is not None and int(rank) != 0:
        return
    try:
        import torch

        target = model.module if hasattr(model, "module") else model
        out = cfg.run.data_dir / f"step3_{cfg.run.run_name}_model.pt"
        torch.save(target.state_dict(), out)
    except Exception:
        pass


def run_training(
    cfg: TrainingConfig,
    ctx: RuntimeContext,
    model: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    train_loader: Any,
    val_loader: Any,
    train_sampler: Any,
    val_sampler: Any,
    monitor: MetricsMonitor,
    proxies: dict,
) -> None:
    import time as _time

    timers = monitor.attach_timers(ctx.device, enabled=bool(cfg.instr.measure_phases))

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
            train_loss, n_train, iter_ms, f_s, b_s, o_s, t_train = train_epoch_fsdp(
                model,
                criterion,
                optimizer,
                scheduler,
                ctx.device,
                train_loader,
                timers=timers,
                progress=progress,
                task_id=task,
            )

        loss_sum, correct, n_val = evaluate(
            model, criterion, ctx.device, val_loader, ctx
        )

        local_elapsed = _time.perf_counter() - _epoch_start
        global_elapsed, global_samples, global_throughput = _global_epoch_reductions(
            local_elapsed, n_train, ctx
        )
        if ctx.rank == 0:
            g_loss = float(loss_sum) / max(1.0, float(n_val))
            g_acc = float(correct) / max(1.0, float(n_val))
        else:
            g_loss = None
            g_acc = None
        speedup, efficiency = _speedup_efficiency(
            cfg, ctx, global_elapsed, global_throughput
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
            fsdp_param_total_bytes=proxies.get("fsdp_param_total_bytes"),
            fsdp_shard_bytes_per_rank=proxies.get("fsdp_shard_bytes_per_rank"),
            fsdp_opt_state_bytes=proxies.get("fsdp_opt_state_bytes"),
            fsdp_expected_comm_per_iter_bytes=proxies.get(
                "fsdp_expected_comm_per_iter_bytes"
            ),
            fsdp_wrapped_modules=proxies.get("fsdp_wrapped_modules"),
            fsdp_sharding_strategy=proxies.get("fsdp_sharding_strategy"),
            global_epoch_sec=global_elapsed,
            global_samples=global_samples,
            global_throughput=global_throughput,
            speedup=speedup,
            efficiency=efficiency,
        )

    monitor.finalize()
    monitor.save_json()
    monitor.plot()
    _save_model("step3", cfg, model, rank=ctx.rank)


def _global_epoch_reductions(local_elapsed: float, n_train: int, ctx: RuntimeContext):
    import torch
    import torch.distributed as dist

    t_elapsed = torch.tensor([local_elapsed], dtype=torch.float64, device="cpu")
    t_samples = torch.tensor([float(n_train)], dtype=torch.float64, device="cpu")
    if ctx.world_size > 1:
        dist.all_reduce(t_elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(t_samples, op=dist.ReduceOp.SUM)
    global_elapsed = float(t_elapsed.item())
    global_samples = int(t_samples.item())
    global_throughput = (
        (global_samples / global_elapsed) if global_elapsed > 0 else None
    )
    return global_elapsed, global_samples, global_throughput


def _speedup_efficiency(
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


def train_epoch_fsdp(
    model,
    criterion,
    optimizer,
    scheduler,
    device,
    train_loader,
    *,
    timers,
    progress,
    task_id: int,
):
    import time as _time

    running_loss = 0.0
    n = 0
    iter_ms: list[float] = []
    for i, (xb, yb) in enumerate(train_loader, start=1):
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
        running_loss += float(loss.item()) * bs
        n += bs
        iter_ms.append((_time.perf_counter() - t_iter0) * 1000.0)
        progress.advance(task_id, 1)
        progress.update(
            task_id,
            description=f"step {i}: loss={float(loss.item()):.4f} it={iter_ms[-1]:.0f}ms",
        )
    f_s = float(getattr(timers, "forward_sec", 0.0))
    b_s = float(getattr(timers, "backward_sec", 0.0))
    o_s = 0.0
    t_epoch_train = f_s + b_s + o_s
    return (running_loss / max(1, n), n, iter_ms, f_s, b_s, o_s, t_epoch_train)


def evaluate(
    model, criterion, device, val_loader, ctx: RuntimeContext
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
            loss_sum += float(loss.item()) * bs
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            n += bs
    if ctx.world_size > 1:
        metrics = torch.tensor(
            [loss_sum, float(correct), float(n)], device="cpu", dtype=torch.float64
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        loss_sum = float(metrics[0].item())
        correct = int(metrics[1].item())
        n = int(metrics[2].item())
    return loss_sum, correct, n


def main(argv: list[str] | None = None) -> int:
    cfg = TrainingConfig.from_argv(argv)
    cfg.apply_seeds()
    ctx, base_model, criterion = setup_distributed_training(cfg)
    model, optimizer, wrapped_count = create_fsdp_model(base_model, cfg, ctx)
    train_loader, val_loader, train_sampler, val_sampler = prepare_data(
        cfg, ctx.rank, ctx.world_size
    )
    scheduler = prepare_scheduler(cfg, optimizer, len(train_loader))
    monitor, proxies = setup_monitoring(cfg, ctx, base_model, model, wrapped_count)
    display_training_header(cfg, ctx, proxies)
    run_training(
        cfg,
        ctx,
        model,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        train_sampler,
        val_sampler,
        monitor,
        proxies,
    )
    monitor.finalize()
    monitor.save_json()
    monitor.plot()
    _save_model("step3", cfg, model, rank=ctx.rank)
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
