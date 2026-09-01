"""Network config selecting the conformal arm, the counterpart of `model_tracking_gatr.py`.

Pass it to their trainer exactly where the projective one goes:

    torchrun --nproc_per_node=4 -m src.train_lightning \
      --network-config src/models/wrapper/model_tracking_cgatr.py \
      ... everything else unchanged ...

Identical to the projective wrapper apart from the import, which is the point: the
comparison is only readable as an algebra result if nothing else moves. See
`src/models/Cgatr_withModifications.py` for what does necessarily differ and why.
"""

import torch

from src.models.Cgatr_withModifications import ExampleWrapper


class GraphTransformerNetWrapper(torch.nn.Module):
    def __init__(self, args, dev, **kwargs) -> None:
        super().__init__()
        self.mod = ExampleWrapper(args, **kwargs)

    def forward(self, g):
        return self.mod(g)


def get_model(data_config, args, dev, **kwargs):
    print("Model options: ", kwargs)
    model = GraphTransformerNetWrapper(args, dev, **kwargs)

    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
            **{"softmax": {0: "N"}},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):

    return torch.nn.MSELoss()
