"""Geometric attention layer for CGA Cl(4,1)."""

from functools import partial

import torch
from torch import nn

from src.cgatr.layers.attention.config import SelfAttentionConfig
from src.cgatr.primitives.attention import geometric_attention, lin_square_normalizer


class GeometricAttention(nn.Module):
    """CGA geometric attention with distance-aware features.

    Parameters
    ----------
    basis_q, basis_k : torch.Tensor with shape (5, 5, 6)
        Distance basis tensors.
    config : SelfAttentionConfig
    """

    def __init__(self, basis_q, basis_k, config: SelfAttentionConfig) -> None:
        super().__init__()
        self.normalizer = partial(lin_square_normalizer, epsilon=config.normalizer_eps)
        self.log_weights = nn.Parameter(
            torch.zeros((config.num_heads, 1, config.hidden_mv_channels))
        )
        self.geometric_attention = geometric_attention(basis_k, basis_q)

    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        weights = self.log_weights.exp()
        h_mv, h_s = self.geometric_attention(
            q_mv, k_mv, v_mv, q_s, k_s, v_s,
            normalizer=self.normalizer,
            weights=weights,
            attn_mask=attention_mask,
        )
        return h_mv, h_s
