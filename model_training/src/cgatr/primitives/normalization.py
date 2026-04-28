"""Equivariant normalization for CGA Cl(4,1)."""

import torch

from src.cgatr.primitives.invariants import inner_product


def equi_layer_norm(
    ip_weights: torch.Tensor,
    x: torch.Tensor,
    channel_dim: int = -2,
    gain: float = 1.0,
    epsilon: float = 0.01,
) -> torch.Tensor:
    """Equivariant LayerNorm using CGA inner product.

    Rescales input such that mean_channels |x|^2 = 1.

    Parameters
    ----------
    ip_weights : torch.Tensor with shape (32,)
        Inner product weights.
    x : torch.Tensor with shape (batch, channels, 32)
    channel_dim : int
    gain : float
    epsilon : float

    Returns
    -------
    outputs : torch.Tensor with shape (batch, channels, 32)
    """
    squared_norms = inner_product(ip_weights, x, x)  # (..., 1)
    # Take absolute value since CGA inner product can be negative
    squared_norms = torch.abs(squared_norms)
    squared_norms = torch.mean(squared_norms, dim=channel_dim, keepdim=True)
    squared_norms = torch.clamp(squared_norms, epsilon)
    outputs = gain * x / torch.sqrt(squared_norms)
    return outputs
