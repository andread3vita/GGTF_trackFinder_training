import argparse
import torch
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.join(script_dir, "..")
working_dir = os.path.abspath(working_dir)
sys.path.append(os.path.join(working_dir, "model_training"))

print(os.path.join(working_dir, "model_training"))


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

from src.models.Gatr_onnx import ExampleWrapper as GravnetModel

parser = argparse.ArgumentParser()
#parser arguments
parser.add_argument("-w", "--weightpath", type=str, default="", help="path to weight file")
parser.add_argument("-o", "--outputpath", type=str, default="", help="path to output directory")
    
original_sparse_coo = torch.sparse_coo_tensor

def debug_sparse_coo(*args, **kwargs):
    print("torch.sparse_coo_tensor was called!")
    import traceback
    traceback.print_stack(limit=5)
    return original_sparse_coo(*args, **kwargs)

torch.sparse_coo_tensor = debug_sparse_coo

args = parser.parse_args()
filepath = args.outputpath + "trackFinder_GATR.onnx"
model_weights = args.weightpath
torch._dynamo.config.verbose = True
model = GravnetModel.load_from_checkpoint(
                    model_weights,
                    args=args,
                    dev='cpu',  
                    map_location=torch.device("cpu")  
                )
model.eval()

args1 = torch.randn((10, 7))
export_options = torch.onnx.ExportOptions(dynamic_shapes=True)


# Export
torch.onnx.export(
    model,
    args1,
    filepath,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,  
    do_constant_folding=True,
    dynamic_axes={"input": [0]},
    verbose=True
)