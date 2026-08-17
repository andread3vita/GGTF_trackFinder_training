import math
from pathlib import Path

import torch
from torch import nn

from src.gatr_v111.interface.rotation import embed_rotation
from src.gatr_v111.layers.linear import EquiLinear
from src.gatr_v111.primitives.linear import (
    _compute_pin_equi_linear_basis,
    reverse,
)
from src.models.equivariant_io import EquivariantBatchNorm1d


def _rotation_z(theta, dtype=torch.float64):
    c, s = math.cos(theta), math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=dtype
    )


def test_isotropic_batchnorm_commutes_with_rigid_motions():
    gen = torch.Generator().manual_seed(7)
    points = torch.randn(128, 3, generator=gen, dtype=torch.float64) * 1000.0
    rotation = _rotation_z(0.73)
    translation = torch.tensor([210.0, -90.0, 37.0], dtype=torch.float64)
    normalizer = EquivariantBatchNorm1d(
        track_running_stats=False
    ).double().train()

    outputs = normalizer(points)
    moved_outputs = normalizer(points @ rotation.T + translation)
    variance = (points - points.mean(dim=0, keepdim=True)).square().mean()
    translated = (
        normalizer.weight * translation / torch.sqrt(variance + normalizer.eps)
    )
    torch.testing.assert_close(
        moved_outputs,
        outputs @ rotation.T + translated,
        atol=1e-12,
        rtol=1e-12,
    )


def test_per_axis_batchnorm_does_not_commute_with_rotations():
    """Control test documenting the defect replaced by fixed scaling."""
    gen = torch.Generator().manual_seed(11)
    points = torch.randn(4096, 3, generator=gen)
    points = points * torch.tensor([1.0, 4.0, 13.0])
    rotation = _rotation_z(0.73, dtype=points.dtype)
    batchnorm = nn.BatchNorm1d(3, affine=False, track_running_stats=False)

    original = batchnorm(points)
    rotated = batchnorm(points @ rotation.T)
    expected = original @ rotation.T
    relative_error = (
        (rotated - expected).abs().max() / expected.abs().max()
    ).item()
    assert relative_error > 0.1


def test_isotropic_running_statistics_preserve_rotations():
    gen = torch.Generator().manual_seed(13)
    points = torch.randn(256, 3, generator=gen, dtype=torch.float64)
    rotation = _rotation_z(0.41)
    normalizer = EquivariantBatchNorm1d().double()
    normalizer.train()
    normalizer(points)
    normalizer.eval()

    outputs = normalizer(points)
    rotated_outputs = normalizer(points @ rotation.T)
    torch.testing.assert_close(
        rotated_outputs,
        outputs @ rotation.T,
        atol=1e-12,
        rtol=1e-12,
    )


def test_gatr_scalar_output_path_is_rotation_invariant():
    """EquiLinear scalar outputs are valid invariant OC coordinates."""
    root = Path(__file__).resolve().parents[1]
    gp = torch.load(
        root / "gatr_utils/geometric_product.pt",
        map_location="cpu",
        weights_only=False,
    ).to_dense().double()
    basis = _compute_pin_equi_linear_basis(dtype=torch.float64)
    layer = EquiLinear(
        basis,
        in_mv_channels=4,
        out_mv_channels=1,
        in_s_channels=None,
        out_s_channels=4,
    ).double()

    gen = torch.Generator().manual_seed(19)
    multivectors = torch.randn(
        32, 4, 16, generator=gen, dtype=torch.float64
    )
    theta = 0.61
    quaternion = torch.tensor(
        [0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)],
        dtype=torch.float64,
    )
    rotor = embed_rotation(quaternion)
    rotor_reverse = reverse(rotor)
    moved = torch.einsum("ijk,j,...k->...i", gp, rotor, multivectors)
    moved = torch.einsum("ijk,...j,k->...i", gp, moved, rotor_reverse)

    _, scalar_outputs = layer(multivectors)
    _, moved_scalar_outputs = layer(moved)
    torch.testing.assert_close(
        moved_scalar_outputs,
        scalar_outputs,
        atol=1e-10,
        rtol=1e-10,
    )
