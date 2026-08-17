#!/usr/bin/env python
"""Measure whether the complete tracking wrapper is invariant to a rigid motion.

This is an architecture check on a fixed-seed, randomly initialized model. It
does not measure tracking accuracy.
"""

import argparse
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.Gatr_onnx import ExampleWrapper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument(
        "--assert-max",
        type=float,
        default=None,
        help="Exit non-zero if the relative residual exceeds this value.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = ExampleWrapper(SimpleNamespace()).eval()
    inputs = torch.randn(args.tokens, 7)
    inputs[:, :3] *= 1000.0
    inputs[:, 3] = torch.randint(0, 2, (args.tokens,), dtype=inputs.dtype)
    inputs[:, 4:] *= 5.0

    angle = 0.71
    c, s = math.cos(angle), math.sin(angle)
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

    max_difference = (outputs - moved_outputs).abs().max().item()
    output_scale = outputs.abs().max().clamp_min(1e-9).item()
    relative_residual = max_difference / output_scale
    print(f"max_abs_difference={max_difference:.9e}")
    print(f"max_abs_output={output_scale:.9e}")
    print(f"relative_rigid_motion_residual={relative_residual:.9e}")

    if args.assert_max is not None and relative_residual > args.assert_max:
        raise SystemExit(
            f"residual {relative_residual:.3e} exceeds {args.assert_max:.3e}"
        )


if __name__ == "__main__":
    main()
