"""
Shared helpers for teaching scripts: CIFAR-10 transforms and dataset creation.

- Prefer using an already-downloaded CIFAR-10 from DATA_DIR.
- Do not auto-download by default to keep runs offline-friendly.
- Callers decide splitting/shuffling; this module only returns datasets.
"""

from __future__ import annotations
import os
import sys


def device_str(force_device=None) -> str:
    if force_device is not None:
        return force_device
    try:
        import torch

        d = os.getenv("DEVICE")
        if d:
            return d
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def init_dist() -> tuple[int, int, str]:
    import torch.distributed as dist

    rank = int(os.getenv("RANK", "-1"))
    world = int(os.getenv("WORLD_SIZE", "-1"))
    backend = os.getenv("BACKEND")
    if backend is None:
        backend = "gloo"
    if rank < 0 or world < 2:
        print(
            "This script requires DDP with >=2 processes. Set RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT.",
            file=sys.stderr,
        )
        sys.exit(2)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size(), str(dist.get_backend())


def cifar_transforms():
    """Return standard CIFAR-10 train/val transforms used across steps."""
    from torchvision import transforms

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
    return tx_train, tx_val


def cifar10_datasets(data_dir: str, *, download_if_missing: bool = False):
    """
    Construct CIFAR-10 train/val datasets with shared transforms.

    Attempts to load without downloading. If `download_if_missing` is True and the
    dataset is not present, it downloads; otherwise it re-raises so callers can
    provide a synthetic fallback.
    """
    from torchvision import datasets

    tx_train, tx_val = cifar_transforms()
    try:
        train_ds = datasets.CIFAR10(
            root=data_dir, train=True, download=False, transform=tx_train
        )
        val_ds = datasets.CIFAR10(
            root=data_dir, train=False, download=False, transform=tx_val
        )
        return train_ds, val_ds
    except Exception:
        if download_if_missing:
            train_ds = datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=tx_train
            )
            val_ds = datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=tx_val
            )
            return train_ds, val_ds
        raise


def synthetic_cifar_like_datasets(
    n_train: int = 512, n_val: int = 256, *, seed: int | None = 0
):
    """Return small synthetic datasets shaped like CIFAR-10 (C=3, H=W=32).

    Used as an offline fallback when CIFAR-10 isn't available locally.
    """
    import torch
    from torch.utils.data import TensorDataset

    if seed is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass
    x_train = torch.randn(n_train, 3, 32, 32)
    y_train = torch.randint(0, 10, (n_train,))
    x_val = torch.randn(n_val, 3, 32, 32)
    y_val = torch.randint(0, 10, (n_val,))
    return TensorDataset(x_train, y_train), TensorDataset(x_val, y_val)


def load_dataset(data_dir: str):
    """Unified dataset getter for teaching steps.

    Tries CIFAR-10 from `data_dir` without downloading. If unavailable (or torchvision
    is missing), returns a small synthetic CIFAR-like dataset.
    """
    try:
        return cifar10_datasets(data_dir, download_if_missing=False)
    except Exception:
        return synthetic_cifar_like_datasets()
