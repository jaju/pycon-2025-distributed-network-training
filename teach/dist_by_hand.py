from __future__ import annotations

import copy
import logging
from typing import Tuple

import numpy as np

from teach import dist_primitives as core

logging.basicConfig(level="DEBUG")


# Create a simple model: 4 layers, input dim=2, hidden dim=6
#
# Model structure (why grads have keys "a" and "b"):
# - Each layer ("block") consists of two linear matrices:
#   - "a": shape (in_dim, hidden_dim)
#   - "b": shape (hidden_dim, in_dim)
# - Forward per layer: x = x @ a; then x = x @ b (two matmuls per block).
# - Backward computes parameter gradients for both matrices and stores them as
#   model.grads[idx]["a"] and model.grads[idx]["b"].
# - In data-parallel training, every rank holds a replica; to match the single
#   process result, we SUM gradients across ranks for both "a" and "b" in every
#   layer (see synchronize_gradients). This keeps replicas in sync and mirrors
#   the full-batch gradient.
base_model = core.Model(num_layers=4, in_dim=2, hidden_dim=6)
# Create some input/target samples
inputs, targets = core.gen_inp(num_samples=10, in_dim=2)


class ManualDDPSyncRunner(core.Runner):
    """Manual, by-hand DDP-style gradient synchronization demo.

    Each rank holds a local replica, runs forward/backward on its shard, then
    explicitly all-reduces per-parameter gradients across peers.
    """

    def partition_global_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return this rank's shard of the global batch (inputs, targets)."""
        # TODO: Implement this section

    def forward_and_loss_grad(
        self, model: core.Model, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Run local forward and compute simple output loss gradient (out - y)."""
        # TODO: Implement this section

    def local_backward(self, model: core.Model, loss_grad: np.ndarray) -> None:
        """Run backward using the provided loss gradient."""
        # TODO: Implement this section

    def synchronize_gradients(self, model: core.Model) -> None:
        """All-reduce SUM each parameter's gradient across ranks."""
        # TODO: Implement this section

    def run(self) -> None:
        # 1) Local model replica per rank
        local_model = copy.deepcopy(self.m)

        # 2) Partition the global batch
        x_local, y_local = self.partition_global_batch()

        # 3) Local forward + simple output loss gradient
        loss_grad = self.forward_and_loss_grad(local_model, x_local, y_local)

        # 4) Backward
        self.local_backward(local_model, loss_grad)

        # 5) Gradient synchronization via all_reduce (SUM)
        self.synchronize_gradients(local_model)

        # 6) Rank 0: inspect and publish results
        if self.rank == 0:
            print("From DDP master process:")
            local_model.print_grads()
            self.publish_model(local_model)


if __name__ == "__main__":
    # Run the distributed job (2 ranks)
    m_final = core.run_world(2, ManualDDPSyncRunner, base_model, inputs, targets)

    # Baseline single-process for comparison
    print("Output from simple single process run:")
    out = base_model.forward(inputs)
    loss = out - targets
    base_model.backward(loss)
    base_model.print_grads()

    print("Comparison:", base_model.compare(m_final))
