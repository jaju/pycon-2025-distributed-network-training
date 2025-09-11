from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import json
import math
import statistics
import time
from contextlib import contextmanager
import os

from rich.console import Console
from rich.table import Table


_console = Console(width=120, force_terminal=True, color_system="standard")


@dataclass(slots=True)
class EpochRecord:
    epoch: int
    train_loss: float | None
    val_loss: float | None
    val_acc: float | None
    epoch_sec: float
    throughput: float | None
    lr: float | None
    # Timing decomposition
    forward_sec: float | None
    backward_sec: float | None
    optimizer_sec: float | None
    total_train_sec: float | None
    # Iteration stats (ms)
    iter_ms_avg: float | None
    iter_ms_p50: float | None
    iter_ms_p90: float | None
    iter_ms_std: float | None
    # DDP metrics (measured or proxies)
    ddp_num_buckets: int | None
    ddp_avg_bucket_size_mb: float | None
    ddp_bucket_time_ms_avg: float | None
    ddp_bucket_time_ms_p50: float | None
    ddp_comm_overhead_s: float | None
    ddp_expected_bucket_count: int | None
    ddp_grad_total_bytes: int | None
    ddp_min_comm_volume_bytes_per_iter: float | None
    # Manual distributed (Step 1)
    comm_overhead_s: float | None
    num_allreduces: int | None
    # FSDP proxies/metrics
    fsdp_param_total_bytes: int | None
    fsdp_shard_bytes_per_rank: float | None
    fsdp_opt_state_bytes: int | None
    fsdp_expected_comm_per_iter_bytes: float | None
    fsdp_wrapped_modules: int | None
    fsdp_sharding_strategy: str | None
    # Optional global aggregates (rank 0)
    global_epoch_sec: float | None = None
    global_throughput: float | None = None
    comm_overhead_global_s: float | None = None
    global_samples: int | None = None
    # Efficiency
    speedup: float | None = None
    efficiency: float | None = None
    # System metrics
    cpu_percent: float | None = None
    ram_gb: float | None = None


@dataclass(slots=True)
class MetricsMonitor:
    run_name: str
    artifacts_dir: Path
    runs_dir: Path
    world_size: int = 1
    rank: int = 0
    show_deltas: bool = True

    _epoch_start_time: float = field(init=False, default=0.0)
    _records: list[EpochRecord] = field(init=False, default_factory=list)
    _tb_writer: Any | None = field(init=False, default=None)
    _ddp_state: dict[str, Any] | None = field(init=False, default=None)
    _timers: Any | None = field(init=False, default=None)
    _psutil_proc: Any | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore

            tbdir = self.runs_dir / "reference" / self.run_name / f"rank{self.rank}"
            tbdir.mkdir(parents=True, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=str(tbdir))
        except Exception:
            self._tb_writer = None
        if self.rank == 0:
            try:
                import psutil  # type: ignore

                self._psutil_proc = psutil.Process(os.getpid())
                try:
                    self._psutil_proc.cpu_percent(None)
                except Exception:
                    pass
            except Exception:
                self._psutil_proc = None

    def epoch_start(self) -> None:
        self._epoch_start_time = time.perf_counter()
        if self._ddp_state is not None:
            self._ddp_state["times"].clear()
            self._ddp_state["num_buckets"] = 0
            self._ddp_state["total_bytes"] = 0
        if self._timers is not None and hasattr(self._timers, "reset_epoch"):
            self._timers.reset_epoch()

    # DDP attachment (optional)
    def attach_ddp(self, ddp_model: Any) -> None:
        try:
            from torch.distributed.algorithms.ddp_comm_hooks import (
                default_hooks as ddp_default_hooks,
            )
            import time as _t

            state = {"times": [], "num_buckets": 0, "total_bytes": 0}

            def timed_allreduce(state_unused, bucket):  # type: ignore[no-untyped-def]
                start = _t.perf_counter()
                buf = bucket.buffer()
                state["total_bytes"] += int(buf.numel()) * int(buf.element_size())
                fut = ddp_default_hooks.allreduce_hook(state_unused, bucket)

                def _done(fut):  # type: ignore[no-untyped-def]
                    state["num_buckets"] += 1
                    state["times"].append(_t.perf_counter() - start)
                    return fut.value()

                return fut.then(_done)

            ddp_model.register_comm_hook(state=None, hook=timed_allreduce)
            self._ddp_state = state
        except Exception:
            self._ddp_state = None

    def local_ddp_comm_overhead_s(self) -> float:
        if self._ddp_state is None:
            return 0.0
        return float(sum(self._ddp_state.get("times", [])))

    def _ddp_epoch_stats(
        self,
    ) -> tuple[int | None, float | None, list[float] | None, float | None]:
        if self._ddp_state is None or not self._ddp_state.get("times"):
            return None, None, None, None
        times = list(self._ddp_state["times"])  # seconds
        num = int(self._ddp_state.get("num_buckets", 0))
        total_bytes = int(self._ddp_state.get("total_bytes", 0))
        avg_mb = (total_bytes / max(1, num)) / 1e6 if num > 0 else None
        return num or None, avg_mb, [t * 1000.0 for t in times], float(sum(times))

    # Phase timers (optional)
    def attach_timers(self, device: Any, enabled: bool) -> Any:
        try:
            self._timers = _PhaseTimers(device, enabled)
        except Exception:
            self._timers = _NullTimers()
        return self._timers

    def epoch_end(
        self,
        *,
        epoch: int,
        train_loss: float | None,
        val_loss: float | None,
        val_acc: float | None,
        num_samples: int | None,
        lr: float | None = None,
        # Timing decomposition
        forward_sec: float | None = None,
        backward_sec: float | None = None,
        optimizer_sec: float | None = None,
        total_train_sec: float | None = None,
        # Iteration stats (ms)
        iter_ms: list[float] | None = None,
        # DDP metrics (measured/hook or proxies)
        ddp_num_buckets: int | None = None,
        ddp_avg_bucket_size_mb: float | None = None,
        ddp_bucket_time_ms: list[float] | None = None,
        ddp_comm_overhead_s: float | None = None,
        ddp_expected_bucket_count: int | None = None,
        ddp_grad_total_bytes: int | None = None,
        ddp_min_comm_volume_bytes_per_iter: float | None = None,
        # Manual distributed (Step 1)
        comm_overhead_s: float | None = None,
        num_allreduces: int | None = None,
        # FSDP proxies
        fsdp_param_total_bytes: int | None = None,
        fsdp_shard_bytes_per_rank: float | None = None,
        fsdp_opt_state_bytes: int | None = None,
        fsdp_expected_comm_per_iter_bytes: float | None = None,
        fsdp_wrapped_modules: int | None = None,
        fsdp_sharding_strategy: str | None = None,
        # Global aggregates (rank 0)
        global_epoch_sec: float | None = None,
        global_samples: int | None = None,
        global_throughput: float | None = None,
        comm_overhead_global_s: float | None = None,
        # Efficiency
        speedup: float | None = None,
        efficiency: float | None = None,
    ) -> None:
        elapsed = max(1e-9, time.perf_counter() - self._epoch_start_time)
        throughput = (
            float(num_samples) / elapsed if (num_samples and num_samples > 0) else None
        )
        # System metrics (rank 0 only)
        cpu_percent: float | None = None
        ram_gb: float | None = None
        if self.rank == 0:
            try:
                import psutil  # type: ignore

                if self._psutil_proc is None:
                    self._psutil_proc = psutil.Process(os.getpid())
                cpu_percent = float(self._psutil_proc.cpu_percent(None))
                rss = float(self._psutil_proc.memory_info().rss)
                ram_gb = rss / (1024.0**3)
            except Exception:
                cpu_percent = None
                ram_gb = None
        # Iteration stats
        if iter_ms and len(iter_ms) > 0:
            iter_avg = float(sum(iter_ms) / len(iter_ms))
            iter_p50 = float(statistics.median(iter_ms))
            iter_ms_sorted = sorted(iter_ms)
            idx = min(len(iter_ms_sorted) - 1, int(0.9 * len(iter_ms_sorted)))
            iter_p90 = float(iter_ms_sorted[idx])
            iter_std = float(statistics.pstdev(iter_ms)) if len(iter_ms) > 1 else 0.0
        else:
            iter_avg = iter_p50 = iter_p90 = iter_std = None

        # DDP hook aggregates (if attached)
        if ddp_bucket_time_ms is None:
            num_buck, avg_mb, bucket_ms, comm_s = self._ddp_epoch_stats()
            if ddp_num_buckets is None:
                ddp_num_buckets = num_buck
            if ddp_avg_bucket_size_mb is None:
                ddp_avg_bucket_size_mb = avg_mb
            if ddp_bucket_time_ms is None:
                ddp_bucket_time_ms = bucket_ms
            if ddp_comm_overhead_s is None:
                ddp_comm_overhead_s = comm_s

        # Prepare record
        rec = EpochRecord(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            epoch_sec=elapsed,
            throughput=throughput,
            lr=lr,
            forward_sec=forward_sec,
            backward_sec=backward_sec,
            optimizer_sec=optimizer_sec,
            total_train_sec=total_train_sec,
            iter_ms_avg=(iter_avg if iter_ms else None),
            iter_ms_p50=(iter_p50 if iter_ms else None),
            iter_ms_p90=(iter_p90 if iter_ms else None),
            iter_ms_std=(iter_std if iter_ms else None),
            ddp_num_buckets=ddp_num_buckets,
            ddp_avg_bucket_size_mb=ddp_avg_bucket_size_mb,
            ddp_bucket_time_ms_avg=(
                (sum(ddp_bucket_time_ms) / len(ddp_bucket_time_ms))
                if ddp_bucket_time_ms
                else None
            ),
            ddp_bucket_time_ms_p50=(
                (statistics.median(ddp_bucket_time_ms)) if ddp_bucket_time_ms else None
            ),
            ddp_comm_overhead_s=ddp_comm_overhead_s,
            ddp_expected_bucket_count=ddp_expected_bucket_count,
            ddp_grad_total_bytes=ddp_grad_total_bytes,
            ddp_min_comm_volume_bytes_per_iter=ddp_min_comm_volume_bytes_per_iter,
            comm_overhead_s=comm_overhead_s,
            num_allreduces=num_allreduces,
            fsdp_param_total_bytes=fsdp_param_total_bytes,
            fsdp_shard_bytes_per_rank=fsdp_shard_bytes_per_rank,
            fsdp_opt_state_bytes=fsdp_opt_state_bytes,
            fsdp_expected_comm_per_iter_bytes=fsdp_expected_comm_per_iter_bytes,
            fsdp_wrapped_modules=fsdp_wrapped_modules,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
            global_epoch_sec=global_epoch_sec,
            global_throughput=global_throughput,
            comm_overhead_global_s=comm_overhead_global_s,
            global_samples=global_samples,
            speedup=speedup,
            efficiency=efficiency,
            cpu_percent=cpu_percent,
            ram_gb=ram_gb,
        )
        self._records.append(rec)

        # Console summary (rank 0)
        if self.rank == 0:
            parts: list[str] = [
                f"[bold cyan]epoch {epoch:03d}[/]",
                f"[yellow]⏱ time[/]=[bold white]{elapsed:.2f}s[/]",
            ]
            prev = self._records[-2] if len(self._records) > 1 else None
            if train_loss is not None:
                seg = f"[yellow]📉 train[/]=[bold white]{train_loss:.4f}[/]"
                if self.show_deltas and prev and prev.train_loss is not None:
                    d = train_loss - prev.train_loss
                    arrow = "↓" if d < 0 else ("↑" if d > 0 else "→")
                    color = "green" if d < 0 else ("red" if d > 0 else "white")
                    seg += f" ([{color}]{arrow}{abs(d):.3f}[/])"
                parts.append(seg)
            if val_loss is not None:
                seg = f"[yellow]📉 val[/]=[bold white]{val_loss:.4f}[/]"
                if self.show_deltas and prev and prev.val_loss is not None:
                    d = val_loss - prev.val_loss
                    arrow = "↓" if d < 0 else ("↑" if d > 0 else "→")
                    color = "green" if d < 0 else ("red" if d > 0 else "white")
                    seg += f" ([{color}]{arrow}{abs(d):.3f}[/])"
                parts.append(seg)
            if val_acc is not None:
                seg = f"[yellow]📈 acc[/]=[bold green]{val_acc * 100:.2f}%[/]"
                if self.show_deltas and prev and prev.val_acc is not None:
                    d = (val_acc - prev.val_acc) * 100.0
                    arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
                    color = "green" if d > 0 else ("red" if d < 0 else "white")
                    seg += f" ([{color}]{arrow}{abs(d):.2f}pp[/])"
                parts.append(seg)
            if throughput is not None:
                parts.append(
                    f"[yellow]🚀 throughput[/]=[bold white]{throughput:.1f} samples/s[/]"
                )
            # Manual DP metrics (if provided)
            if comm_overhead_s is not None:
                parts.append(
                    f"[yellow]dist/comm[/]=[bold white]{comm_overhead_s:.2f}s[/]"
                )
            if num_allreduces is not None:
                parts.append(
                    f"[yellow]dist/calls[/]=[bold white]{num_allreduces} all_reduce[/]"
                )
            _console.print(" | ".join(parts))

        # TensorBoard scalars
        if self._tb_writer is not None:
            if train_loss is not None:
                self._tb_writer.add_scalar("loss/train", train_loss, epoch)
            if val_loss is not None:
                self._tb_writer.add_scalar("loss/val", val_loss, epoch)
            if val_acc is not None:
                self._tb_writer.add_scalar("accuracy/val", val_acc, epoch)
            self._tb_writer.add_scalar("time/epoch_sec", elapsed, epoch)
            if throughput is not None:
                self._tb_writer.add_scalar(
                    "throughput/samples_per_sec", throughput, epoch
                )
            if lr is not None:
                self._tb_writer.add_scalar("opt/lr", lr, epoch)
            if self.rank == 0:
                if cpu_percent is not None:
                    self._tb_writer.add_scalar("sys/cpu_percent", cpu_percent, epoch)
                if ram_gb is not None:
                    self._tb_writer.add_scalar("sys/ram_gb", ram_gb, epoch)
            # Optional DDP hook stats
            if ddp_num_buckets is not None:
                self._tb_writer.add_scalar("ddp/num_buckets", ddp_num_buckets, epoch)
            if ddp_avg_bucket_size_mb is not None:
                self._tb_writer.add_scalar(
                    "ddp/avg_bucket_size_mb", ddp_avg_bucket_size_mb, epoch
                )
            if ddp_comm_overhead_s is not None:
                self._tb_writer.add_scalar(
                    "ddp/comm_overhead_s", ddp_comm_overhead_s, epoch
                )
            # Optional: DDP hook stats
            if ddp_num_buckets is not None:
                self._tb_writer.add_scalar("ddp/num_buckets", ddp_num_buckets, epoch)
            if ddp_avg_bucket_size_mb is not None:
                self._tb_writer.add_scalar(
                    "ddp/avg_bucket_size_mb", ddp_avg_bucket_size_mb, epoch
                )
            if ddp_comm_overhead_s is not None:
                self._tb_writer.add_scalar(
                    "ddp/comm_overhead_s", ddp_comm_overhead_s, epoch
                )
            # Manual DP metrics
            if comm_overhead_s is not None:
                self._tb_writer.add_scalar(
                    "dist/comm_overhead_s", comm_overhead_s, epoch
                )
            if num_allreduces is not None:
                self._tb_writer.add_scalar("dist/num_allreduces", num_allreduces, epoch)
            # Global aggregates
            if global_epoch_sec is not None:
                self._tb_writer.add_scalar(
                    "global/time/epoch_sec", global_epoch_sec, epoch
                )
            if global_throughput is not None:
                self._tb_writer.add_scalar(
                    "global/throughput/samples_per_sec", global_throughput, epoch
                )
            if comm_overhead_global_s is not None:
                self._tb_writer.add_scalar(
                    "global/dist/comm_overhead_s", comm_overhead_global_s, epoch
                )
            if global_samples is not None:
                self._tb_writer.add_scalar(
                    "global/samples/train", global_samples, epoch
                )
            if speedup is not None:
                self._tb_writer.add_scalar("global/speedup", speedup, epoch)
            if efficiency is not None:
                self._tb_writer.add_scalar("global/efficiency", efficiency, epoch)
            self._tb_writer.add_scalar("dist/world_size", self.world_size, epoch)
            self._tb_writer.flush()

    # Optional: log profiler window aggregates (rank 0 only)
    def log_profiler_metrics(
        self, comm_overhead_s: float, num_collectives: int
    ) -> None:
        if self.rank != 0 or self._tb_writer is None:
            return
        try:
            self._tb_writer.add_scalar(
                "dist/prof_comm_overhead_s", float(comm_overhead_s), 0
            )
            self._tb_writer.add_scalar(
                "dist/prof_num_collectives", int(num_collectives), 0
            )
            self._tb_writer.flush()
        except Exception:
            pass

    def finalize(self) -> None:
        if self.rank != 0:
            return
        table = Table(title=f"Run Summary · {self.run_name}")
        table.add_column("Epoch", justify="right")
        table.add_column("Train Loss", justify="right")
        table.add_column("Val Loss", justify="right")
        table.add_column("Val Acc", justify="right")
        table.add_column("Epoch (s)", justify="right")
        table.add_column("Throughput", justify="right")
        for r in self._records:
            table.add_row(
                str(r.epoch),
                f"{r.train_loss:.4f}" if r.train_loss is not None else "-",
                f"{r.val_loss:.4f}" if r.val_loss is not None else "-",
                f"{r.val_acc * 100:.2f}%" if r.val_acc is not None else "-",
                f"{r.epoch_sec:.2f}",
                f"{r.throughput:.1f}/s" if r.throughput is not None else "-",
            )
        _console.print(table)

    def save_json(self) -> Path | None:
        if self.rank != 0:
            return None
        out = self.artifacts_dir / "reference" / f"{self.run_name}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_name": self.run_name,
            "world_size": self.world_size,
            "epochs": [asdict(r) for r in self._records],
        }
        out.write_text(json.dumps(payload, indent=2))
        return out

    def plot(self) -> Path | None:
        if self.rank != 0:
            return None
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import seaborn as sns  # type: ignore

            sns.set_theme(style="whitegrid")
            fig, ax1 = plt.subplots(figsize=(7.5, 4.0), dpi=150)
            epochs = [r.epoch for r in self._records]
            train = [r.train_loss for r in self._records]
            val = [r.val_loss for r in self._records]
            acc = [r.val_acc for r in self._records]

            if any(v is not None for v in train):
                ax1.plot(
                    epochs,
                    [t if t is not None else math.nan for t in train],
                    label="train loss",
                )
            if any(v is not None for v in val):
                ax1.plot(
                    epochs,
                    [v if v is not None else math.nan for v in val],
                    label="val loss",
                )
            ax1.set_xlabel("epoch")
            ax1.set_ylabel("loss")

            if any(v is not None for v in acc):
                ax2 = ax1.twinx()
                ax2.plot(
                    epochs,
                    [a * 100.0 if a is not None else math.nan for a in acc],
                    label="val acc",
                    color="#2ca02c",
                )
                ax2.set_ylabel("acc (%)")

            ax1.legend(loc="upper right")
            out = self.artifacts_dir / "reference" / f"{self.run_name}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            return out
        except Exception:
            return None


class _PhaseTimers:
    def __init__(self, device: Any, enabled: bool) -> None:
        import time as _t

        self.enabled = bool(enabled)
        self.device_type = getattr(
            getattr(device, "type", None), "lower", lambda: str(device)
        )()
        self._time = _t
        self.forward_sec = 0.0
        self.backward_sec = 0.0
        self.iter_ms: list[float] = []
        self._iter_t0 = 0.0
        self._cuda = None
        if self.enabled and self.device_type == "cuda":
            try:
                import torch

                self._cuda = torch.cuda
                self._f_start = self._cuda.Event(enable_timing=True)
                self._f_end = self._cuda.Event(enable_timing=True)
                self._b_start = self._cuda.Event(enable_timing=True)
                self._b_end = self._cuda.Event(enable_timing=True)
            except Exception:
                self._cuda = None

    def reset_epoch(self) -> None:
        self.forward_sec = 0.0
        self.backward_sec = 0.0
        self.iter_ms.clear()
        self._iter_t0 = 0.0

    def iter_start(self) -> None:
        self._iter_t0 = self._time.perf_counter()

    def iter_end(self) -> None:
        if self._iter_t0:
            self.iter_ms.append((self._time.perf_counter() - self._iter_t0) * 1000.0)
            self._iter_t0 = 0.0

    @contextmanager
    def forward(self):  # type: ignore[no-untyped-def]
        if self.enabled and self._cuda is not None:
            self._f_start.record()
            yield
            self._f_end.record()
            self._f_end.synchronize()
            self.forward_sec += self._f_start.elapsed_time(self._f_end) / 1000.0
        else:
            t0 = self._time.perf_counter()
            yield
            self.forward_sec += self._time.perf_counter() - t0

    @contextmanager
    def backward(self):  # type: ignore[no-untyped-def]
        if self.enabled and self._cuda is not None:
            self._b_start.record()
            yield
            self._b_end.record()
            self._b_end.synchronize()
            self.backward_sec += self._b_start.elapsed_time(self._b_end) / 1000.0
        else:
            t0 = self._time.perf_counter()
            yield
            self.backward_sec += self._time.perf_counter() - t0


class _NullTimers:
    forward_sec: float = 0.0
    backward_sec: float = 0.0
    iter_ms: list[float] = []

    def reset_epoch(self) -> None:
        self.forward_sec = 0.0
        self.backward_sec = 0.0
        self.iter_ms = []

    def iter_start(self) -> None:  # no-op
        return None

    def iter_end(self) -> None:  # no-op
        return None

    @contextmanager
    def forward(self):  # type: ignore[no-untyped-def]
        yield

    @contextmanager
    def backward(self):  # type: ignore[no-untyped-def]
        yield
