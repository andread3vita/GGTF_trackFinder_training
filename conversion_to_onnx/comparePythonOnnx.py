import argparse
import torch
import os
import sys
import onnxruntime as ort
import numpy as np

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(this_dir, "..")
absolute = os.path.abspath(parent_dir)

final_path = os.path.join(absolute, "model_training")
sys.path.append(final_path)

from onnxscript.function_libs.torch_lib.ops import nn as _nn
from onnxscript import opset18 as op

def _gelu_tanh_fix(x):
    # float constant as scalar with value_float
    three = op.Constant(value_float=3.0)
    coeff = op.Constant(value_float=0.044715)
    return op.Mul(op.Pow(x, three), coeff) + x

def _gelu_none_fix(x):
    sqrt2 = op.Constant(value_float=1.4142135)
    return op.Mul(op.Erf(op.Div(x, sqrt2)), x)

_nn._aten_gelu_approximate_tanh = _gelu_tanh_fix
_nn._aten_gelu_approximate_none = _gelu_none_fix


##################################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-w","--weightPath",type=str, default="",help="path to weight file")
parser.add_argument("-o","--onnxPath",type=str, default="/",help="path to onnx file")   
args = parser.parse_args()

torch.manual_seed(42)
input = torch.randn((30, 7), dtype=torch.float32)

from src.models.Gatr_onnx import ExampleWrapper as GravnetModel

model_weights = args.weightPath

torch._dynamo.config.verbose = True

model = GravnetModel.load_from_checkpoint(
    model_weights,
    args=args,
    dev='cpu',
    map_location=torch.device("cpu"))

model.eval()


with torch.no_grad():
    output_torch = model(input)
    
# output from ONNX model
input_np = input.numpy().astype(np.float32)
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
onnx_model_path = args.onnxPath
session = ort.InferenceSession(onnx_model_path, sess_options)
input_name = session.get_inputs()[0].name

outputs = session.run(None, {input_name: input_np})
output_onnx = outputs[0]
output_onnx = torch.tensor(output_onnx, dtype=torch.float32, device='cpu')

print("Output Pure Python (PyTorch):", output_torch)
print("Output ONNX Runtime:", output_onnx)

print("\nPyTorch Output:")
print("  Type:", type(output_torch))
print("  Dtype:", output_torch.dtype)
print("  Shape:", output_torch.shape)

print("\nONNX Output:")
print("  Type:", type(output_onnx))
print("  Dtype:", output_onnx.dtype)
print("  Shape:", output_onnx.shape)

if torch.is_tensor(output_onnx):
    output_onnx_np = output_onnx.detach().cpu().numpy()
else:
    output_onnx_np = np.array(output_onnx)

output_torch_np = output_torch.detach().cpu().numpy()

max_diff = np.max(np.abs(output_torch_np - output_onnx_np))
are_close = np.allclose(output_torch_np, output_onnx_np, rtol=1e-5, atol=1e-6)

print("\nComparison:")
print("  Max Absolute Difference:", max_diff)
print("  Are outputs close?:", are_close)

def get_clustering(betas: torch.Tensor, X: torch.Tensor, tbeta=0.7, td=0.05):
    """
    Returns a clustering of hits -> cluster_index, based on the GravNet model
    output (predicted betas and cluster space coordinates) and the clustering
    parameters tbeta and td.
    Takes torch.Tensors as input.
    """
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    # Assign points to condensation points
    # Only assign previously unassigned points (no overwriting)
    # Points unassigned at the end are bkg (-1)
    unassigned = torch.arange(n_points).to(betas.device)
    clustering = -1 * torch.ones(n_points, dtype=torch.long).to(betas.device)
    while len(indices_condpoints) > 0 and len(unassigned) > 0:
        index_condpoint = indices_condpoints[0]
        d = torch.norm(X[unassigned] - X[index_condpoint][0], dim=-1)
        assigned_to_this_condpoint = unassigned[d < td]
        clustering[assigned_to_this_condpoint] = index_condpoint[0]
        unassigned = unassigned[~(d < td)]
        
        # calculate indices_codpoints again
        indices_condpoints = find_condpoints(betas, unassigned, tbeta)
    return clustering


def find_condpoints(betas, unassigned, tbeta):
    n_points = betas.size(0)
    select_condpoints = betas > tbeta
    device = betas.device
    mask_unassigned = torch.zeros(n_points).to(device)
    mask_unassigned[unassigned] = True
    select_condpoints = mask_unassigned.to(bool) * select_condpoints
    # Get indices passing the threshold
    indices_condpoints = select_condpoints.nonzero()
    # Order them by decreasing beta value
    indices_condpoints = indices_condpoints[(-betas[select_condpoints]).argsort()]
    return indices_condpoints


with torch.no_grad():
    
    betas_torch = output_torch[:, 3]
    X_torch = output_torch[:, 0:3]

# Clustering PyTorch
clustering_torch = get_clustering(betas_torch, X_torch, tbeta=0.6, td=0.3)


betas_onnx = output_onnx[:, 3]
X_onnx = output_onnx[:, 0:3]

# Clustering ONNX
clustering_onnx = get_clustering(betas_onnx, X_onnx, tbeta=0.6, td=0.3)


print("\n\nClustering PyTorch:", clustering_torch)
print("Clustering ONNX:", clustering_onnx)

if torch.equal(clustering_torch, clustering_onnx):
    print("PyTorch and ONNX clusterings are identical")
else:
    print("PyTorch and ONNX clusterings are not identical.")
    for i, (c_t, c_o) in enumerate(zip(clustering_torch, clustering_onnx)):
        if c_t != c_o:
            print(f"Point {i}: PyTorch={c_t.item()}, ONNX={c_o.item()}")

