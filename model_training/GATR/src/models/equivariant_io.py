"""Rotation-equivariant input normalization for the tracking models."""

import torch
from torch import nn


class EquivariantBatchNorm1d(nn.Module):
    """Batch-normalize vectors with one shared variance and gain.

    A per-axis BatchNorm does not commute with rotations: each coordinate gets
    a different variance, gain, and offset. One scalar for the complete vector
    preserves rotations without introducing a detector-specific fixed scale.
    """

    def __init__(
        self,
        momentum: float = 0.1,
        eps: float = 1e-5,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.weight = nn.Parameter(torch.ones(()))
        if track_running_stats:
            self.register_buffer("running_var", torch.ones(()))
        else:
            self.register_buffer("running_var", None)

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.shape[-1] != 3:
            raise ValueError("vectors must have three components")

        use_batch_stats = self.training or not self.track_running_stats
        if use_batch_stats:
            centered = vectors - vectors.mean(dim=0, keepdim=True)
            variance = centered.square().mean()
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.running_var.lerp_(variance.detach(), self.momentum)
        else:
            variance = self.running_var

        return self.weight * vectors / torch.sqrt(variance + self.eps)
