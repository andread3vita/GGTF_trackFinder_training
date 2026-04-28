"""Bilinear operations for CGA Cl(4,1): geometric product and outer product.

All operations use 32-component multivectors and (32, 32, 32) Cayley tables.
"""

import torch
from torch import nn


class geometric_product(nn.Module):
    """Geometric product using precomputed Cayley table."""

    def __init__(self, gp) -> None:
        super().__init__()
        self.register_buffer("gp", gp)  # (32, 32, 32)

    def forward(self, x, y):
        # x, y: (..., batch, channels, 32)
        # Two-step einsum for ONNX compatibility
        outputs1 = torch.einsum("i j k, ab j-> abik", self.gp, x)
        outputs = torch.einsum("abik, abk -> ab i", outputs1, y)
        return outputs


def outer_product(op, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the outer (wedge) product f(x,y) = x ^ y.

    Parameters
    ----------
    op : torch.Tensor with shape (32, 32, 32)
        Outer product Cayley table.
    x : torch.Tensor with shape (..., 32)
        First input multivector.
    y : torch.Tensor with shape (..., 32)
        Second input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 32)
        Wedge product result.
    """
    outputs1 = torch.einsum("i j k, ab j-> abik", op, x)
    outputs = torch.einsum("abik, abk -> ab i", outputs1, y)
    return outputs
