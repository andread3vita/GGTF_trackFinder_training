"""Bilinear operations for CGA Cl(4,1): geometric product and outer product.

All operations use 32-component multivectors and (32, 32, 32) Cayley tables.
"""

import torch
from torch import nn


class _GeometricProductFn(torch.autograd.Function):
    """Geometric product with a memory-frugal backward.

    The naive two-step einsum saves the ``(..., 32, 32)`` intermediate
    ``outputs1`` for backward — i.e. ``N x C x 1024`` floats per call, which
    dominates activation memory for large hit counts. Here we save only the
    two ``(..., 32)`` inputs and compute the (analytic) gradient directly as a
    pair of bilinear contractions, so the 32x-larger intermediate is never
    retained. The forward value is bit-for-bit identical to the two-step
    einsum, so model outputs — and hence equivariance — are unchanged.

        out_i      = sum_jk  G[i,j,k] x_j y_k
        grad_x_j   = sum_ik  G[i,j,k] go_i y_k
        grad_y_k   = sum_ij  G[i,j,k] go_i x_j
    """

    @staticmethod
    def forward(ctx, gp, x, y):
        outputs1 = torch.einsum("ijk, ...j -> ...ik", gp, x)  # transient, not saved
        out = torch.einsum("...ik, ...k -> ...i", outputs1, y)
        ctx.save_for_backward(gp, x, y)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        gp, x, y = ctx.saved_tensors
        grad_x = grad_y = None
        if ctx.needs_input_grad[1]:
            grad_x = torch.einsum("ijk, ...i, ...k -> ...j", gp, grad_out, y)
        if ctx.needs_input_grad[2]:
            grad_y = torch.einsum("ijk, ...i, ...j -> ...k", gp, grad_out, x)
        return None, grad_x, grad_y


class geometric_product(nn.Module):
    """Geometric product using precomputed Cayley table."""

    def __init__(self, gp) -> None:
        super().__init__()
        self.register_buffer("gp", gp)  # (32, 32, 32)

    def forward(self, x, y):
        # x, y: (..., 32). Arbitrary leading dims supported via ellipsis
        # so the same op handles single-event (items, channels, 32) and
        # multi-event batched (B, N, channels, 32) inputs without a
        # custom rank-2 prefix.
        if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
            # Training: memory-frugal custom backward (no (...,32,32) saved).
            return _GeometricProductFn.apply(self.gp, x, y)
        # Inference / ONNX export: plain two-step einsum (traceable, no backward
        # needed). Mathematically identical to the training path.
        outputs1 = torch.einsum("ijk, ...j -> ...ik", self.gp, x)
        outputs = torch.einsum("...ik, ...k -> ...i", outputs1, y)
        return outputs


def outer_product(op, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the outer (wedge) product f(x,y) = x ^ y.

    Parameters
    ----------
    op : torch.Tensor with shape (32, 32, 32)
        Outer product Cayley table.
    x : torch.Tensor with shape (..., 32)
        First input multivector. Arbitrary leading dims supported.
    y : torch.Tensor with shape (..., 32)
        Second input multivector.

    Returns
    -------
    outputs : torch.Tensor with shape (..., 32)
        Wedge product result.
    """
    outputs1 = torch.einsum("ijk, ...j -> ...ik", op, x)
    outputs = torch.einsum("...ik, ...k -> ...i", outputs1, y)
    return outputs
