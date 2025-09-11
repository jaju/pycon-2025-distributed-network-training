from __future__ import annotations

import logging
import multiprocessing as mp
import socket
from contextlib import closing
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

logging.basicConfig(level="INFO")


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Model:
    def __init__(self, num_layers: int, in_dim: int, hidden_dim: int):
        self.n = int(num_layers)
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)

        self.blocks: list[dict[str, np.ndarray]] = []
        self.inputs: list[dict[str, np.ndarray | None]] = []
        self.grads: list[dict[str, np.ndarray]] = []

        for _ in range(self.n):
            self.blocks.append(
                {
                    "a": np.random.rand(self.in_dim, self.hidden_dim),
                    "b": np.random.rand(self.hidden_dim, self.in_dim),
                }
            )
            self.inputs.append({"a": None, "b": None})
            self.grads.append(
                {
                    "a": np.zeros((self.in_dim, self.hidden_dim), dtype=np.float64),
                    "b": np.zeros((self.hidden_dim, self.in_dim), dtype=np.float64),
                }
            )

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i in range(self.n):
            self.inputs[i]["a"] = x
            x = x @ self.blocks[i]["a"]
            self.inputs[i]["b"] = x
            x = x @ self.blocks[i]["b"]
        return x

    def backward(self, loss_grad: np.ndarray) -> None:
        act_grad = loss_grad
        for i in range(self.n - 1, -1, -1):
            self.grads[i]["b"] = self.inputs[i]["b"].T @ act_grad
            act_grad = act_grad @ self.blocks[i]["b"].T
            self.grads[i]["a"] = self.inputs[i]["a"].T @ act_grad
            act_grad = act_grad @ self.blocks[i]["a"].T

    def print_grads(self) -> None:
        print("Grads:")
        for i in range(self.n):
            print(
                f"Layer {i}: a.shape={self.grads[i]['a'].shape}, b.shape={self.grads[i]['b'].shape}"
            )

    def compare(self, other: "Model") -> bool:
        for i in range(self.n):
            if not np.allclose(self.grads[i]["a"], other.grads[i]["a"]):
                logging.debug(f"Mismatch at layer {i}, param a")
                return False
            if not np.allclose(self.grads[i]["b"], other.grads[i]["b"]):
                logging.debug(f"Mismatch at layer {i}, param b")
                return False
        return True


def gen_inp(num_samples: int, in_dim: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(int(num_samples), int(in_dim)).astype(np.float64)
    y = np.random.rand(int(num_samples), int(in_dim)).astype(np.float64)
    return x, y


class TorchCCL:
    def all_reduce(self, arr: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(arr.copy())
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.numpy()


class Runner:
    def __init__(
        self,
        rank: int,
        world_size: int,
        ccl: TorchCCL,
        m: Model,
        i: np.ndarray,
        o: np.ndarray,
        shared: Any,
    ) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.ccl = ccl
        self.m = m
        self.i = i
        self.o = o
        self._shared = shared

    def publish_model(self, model: Model) -> None:
        if self.rank != 0:
            return
        payload: list[dict[str, np.ndarray]] = []
        for idx in range(model.n):
            payload.append({"a": model.grads[idx]["a"], "b": model.grads[idx]["b"]})
        self._shared["ddp_grads"] = payload

    def run(self) -> None:  # pragma: no cover - to be implemented by user
        raise NotImplementedError


def _worker(
    rank: int,
    world_size: int,
    init_method: str,
    runner_cls: type[Runner],
    shared,
    m: Model,
    i: np.ndarray,
    o: np.ndarray,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=int(rank),
        world_size=int(world_size),
    )
    try:
        ccl = TorchCCL()
        runner = runner_cls(rank, world_size, ccl, m, i, o, shared)
        runner.run()
    finally:
        dist.destroy_process_group()


def run_world(
    world_size: int, runner_cls: type[Runner], m: Model, i: np.ndarray, o: np.ndarray
) -> Model:
    mp.set_start_method("spawn", force=True)
    port = _find_free_port()
    init_method = f"tcp://127.0.0.1:{port}"

    with mp.Manager() as manager:
        shared = manager.dict()
        procs: list[mp.Process] = []
        for r in range(int(world_size)):
            p = mp.Process(
                target=_worker,
                args=(r, world_size, init_method, runner_cls, shared, m, i, o),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

        ddp_grads = shared.get("ddp_grads")
        if ddp_grads is None:
            raise RuntimeError("Rank 0 did not publish DDP gradients.")

        m_ddp = Model(num_layers=m.n, in_dim=m.in_dim, hidden_dim=m.hidden_dim)
        for idx in range(m.n):
            m_ddp.grads[idx]["a"] = np.array(ddp_grads[idx]["a"])
            m_ddp.grads[idx]["b"] = np.array(ddp_grads[idx]["b"])
        return m_ddp
