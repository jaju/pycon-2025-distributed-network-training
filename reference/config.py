from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, get_type_hints

import argparse
import os
import random


def _detect_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Core
@dataclass(slots=True)
class RunConfig:
    run_name: str = field(
        default_factory=lambda: f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    seed: int = 42
    deterministic: bool = False
    device: str = field(default="cpu")
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    runs_dir: Path = field(default_factory=lambda: Path(os.getenv("RUNS_DIR", "runs")))
    artifacts_dir: Path = field(
        default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    )


@dataclass(slots=True)
class DataConfig:
    batch_size: int = 128
    synthetic_if_missing: bool = True
    synthetic_train_size: int = 512
    synthetic_val_size: int = 256
    download_data: bool = False


@dataclass(slots=True)
class OptimConfig:
    epochs: int = 1
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    # Scheduler
    scheduler: str = "none"  # none|cosine|step|onecycle
    cosine_tmax: int = 0
    cosine_eta_min: float = 0.0
    step_size: int = 30
    gamma: float = 0.1
    onecycle_max_lr: float = 0.0
    onecycle_pct_start: float = 0.3


@dataclass(slots=True)
class ConsoleConfig:
    show_deltas: bool = True
    progress_bar: bool = True
    # Optional minimal heartbeat on non-zero ranks (used in some steps)
    worker_heartbeat: bool = False
    worker_heartbeat_every: int = 20


# Step 1 extras
@dataclass(slots=True)
class DistConfig:
    broadcast_buffers: bool = False


# Step 2 (DDP)
@dataclass(slots=True)
class DDPConfig:
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    static_graph: bool = False
    gradient_as_bucket_view: bool = True


# Step 3 (FSDP)
@dataclass(slots=True)
class FSDPConfig:
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD|SHARD_GRAD_OP|NO_SHARD
    auto_wrap_threshold: int = 0
    use_orig_params: bool = True
    backward_prefetch: str = "BACKWARD_POST"  # BACKWARD_PRE|BACKWARD_POST
    forward_prefetch: bool = False
    mixed_precision: str = "none"  # none|fp16|bf16
    cpu_offload: bool = False
    limit_all_gathers: bool = True
    activation_checkpointing: bool = False
    simulate: bool = True


@dataclass(slots=True)
class InstrConfig:
    # Shared instrumentation across steps
    measure_phases: bool = False
    profiler_on: bool = False
    profiler_warmup_steps: int = 1
    profiler_active_steps: int = 5
    baseline_throughput: float = 0.0
    baseline_epoch_sec: float = 0.0
    # DDP
    ddp_comm_stats: bool = True
    # FSDP
    fsdp_comm_stats: bool = True


@dataclass(slots=True)
class TrainingConfig:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    console: ConsoleConfig = field(default_factory=ConsoleConfig)
    # Extras used by specific steps
    dist: DistConfig = field(default_factory=DistConfig)
    ddp: DDPConfig = field(default_factory=DDPConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    instr: InstrConfig = field(default_factory=InstrConfig)

    def finalize(self, explicit_device: str | None = None) -> None:
        self.run.device = _detect_device(explicit_device or self.run.device)
        if not isinstance(self.run.data_dir, Path):
            self.run.data_dir = Path(self.run.data_dir)
        if not isinstance(self.run.runs_dir, Path):
            self.run.runs_dir = Path(self.run.runs_dir)
        if not isinstance(self.run.artifacts_dir, Path):
            self.run.artifacts_dir = Path(self.run.artifacts_dir)
        _ensure_dir(self.run.data_dir)
        _ensure_dir(self.run.runs_dir)
        _ensure_dir(self.run.artifacts_dir)
        # Normalize and validate scheduler early
        sched = (self.optim.scheduler or "none").lower()
        allowed = {"none", "cosine", "step", "onecycle"}
        if sched not in allowed:
            raise ValueError(
                f"Unknown scheduler '{self.optim.scheduler}'. Use one of: none, cosine, step, onecycle."
            )
        self.optim.scheduler = sched
        if self.optim.onecycle_max_lr == 0.0:
            self.optim.onecycle_max_lr = self.optim.lr

    @staticmethod
    def _add_fields(
        parser: argparse.ArgumentParser,
        dc_type: type[Any],
        group: argparse._ArgumentGroup | None = None,
    ) -> None:
        hints = get_type_hints(dc_type)
        add_to = group if group is not None else parser
        for f in fields(dc_type):
            name = f.name
            arg_hy = f"--{name.replace('_', '-')}"
            arg_us = f"--{name}"
            kwargs: dict[str, Any] = {}
            default_val = (
                f.default
                if f.default is not MISSING
                else (f.default_factory() if f.default_factory is not MISSING else None)
            )
            ftype = hints.get(name, f.type)
            if ftype is bool:
                g = (group or parser).add_mutually_exclusive_group(required=False)
                g.add_argument(
                    arg_hy, dest=name, action="store_true", help=f"Set {name}=True"
                )
                g.add_argument(
                    f"--no-{name.replace('_', '-')}",
                    dest=name,
                    action="store_false",
                    help=f"Set {name}=False",
                )
                if arg_us != arg_hy:
                    g.add_argument(arg_us, dest=name, action="store_true", help=arg_hy)
                parser.set_defaults(**{name: bool(default_val)})
                continue
            if ftype is Path:
                kwargs["type"] = str
            elif ftype in (int, float, str):
                kwargs["type"] = ftype
            else:
                kwargs["type"] = type(default_val) if default_val is not None else str
            kwargs["default"] = default_val
            kwargs["help"] = f"Default: {default_val!r}"
            add_to.add_argument(arg_hy, **kwargs)
            if arg_us != arg_hy:
                add_to.add_argument(
                    arg_us, **{k: v for k, v in kwargs.items() if k != "help"}
                )

    @classmethod
    def from_argv(cls, argv: Iterable[str] | None = None) -> "TrainingConfig":
        parser = argparse.ArgumentParser(
            description="Unified training configuration (baseline → DDP → FSDP)"
        )
        # Grouped, flattened CLI
        g_run = parser.add_argument_group("Run/Paths")
        cls._add_fields(parser, RunConfig, g_run)
        g_data = parser.add_argument_group("Data")
        cls._add_fields(parser, DataConfig, g_data)
        g_optim = parser.add_argument_group("Optim/Scheduler")
        cls._add_fields(parser, OptimConfig, g_optim)
        g_console = parser.add_argument_group("Console")
        cls._add_fields(parser, ConsoleConfig, g_console)
        g_dist = parser.add_argument_group("Distributed (manual)")
        cls._add_fields(parser, DistConfig, g_dist)
        g_ddp = parser.add_argument_group("DDP")
        cls._add_fields(parser, DDPConfig, g_ddp)
        g_instr = parser.add_argument_group("Instrumentation/Profiler")
        cls._add_fields(parser, InstrConfig, g_instr)
        g_fsdp = parser.add_argument_group("FSDP")
        cls._add_fields(parser, FSDPConfig, g_fsdp)

        args = parser.parse_args(list(argv) if argv is not None else None)
        ns = vars(args)

        def build(dc):
            keys = {f.name for f in fields(dc)}
            return dc(**{k: ns[k] for k in keys if k in ns})

        cfg = cls(
            run=build(RunConfig),
            data=build(DataConfig),
            optim=build(OptimConfig),
            console=build(ConsoleConfig),
            dist=build(DistConfig),
            ddp=build(DDPConfig),
            fsdp=build(FSDPConfig),
            instr=build(InstrConfig),
        )
        cfg.finalize()
        return cfg

    def apply_seeds(self) -> None:
        random.seed(self.run.seed)
        try:
            import numpy as np  # type: ignore

            np.random.seed(self.run.seed)
        except Exception:
            pass
        try:
            import torch  # type: ignore

            torch.manual_seed(self.run.seed)
            if self.run.device == "cuda" and torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.run.seed)
            if self.run.deterministic:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
