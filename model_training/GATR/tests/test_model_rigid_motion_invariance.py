import math
from types import SimpleNamespace

import torch

from src.models.Gatr_onnx import ExampleWrapper


def test_model_outputs_are_invariant_under_rigid_motions():
    torch.manual_seed(3)
    model = ExampleWrapper(SimpleNamespace()).eval()
    inputs = torch.randn(16, 7)
    inputs[:, :3] *= 1000.0
    inputs[:, 3] = torch.randint(0, 2, (16,), dtype=inputs.dtype)
    inputs[:, 4:] *= 5.0

    theta = 0.71
    c, s = math.cos(theta), math.sin(theta)
    rotation = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    )
    translation = torch.tensor([210.0, -90.0, 37.0])
    moved = inputs.clone()
    moved[:, :3] = inputs[:, :3] @ rotation.T + translation
    moved[:, 4:] = inputs[:, 4:] @ rotation.T

    with torch.no_grad():
        outputs = model(inputs)
        moved_outputs = model(moved)

    relative_error = (
        (outputs - moved_outputs).abs().max()
        / outputs.abs().max().clamp_min(1e-9)
    ).item()
    assert relative_error < 2e-5
