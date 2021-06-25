import random
import torch
import torchvision.models as models
import torch.autograd.profiler as profiler
import onnx
from onnx import helper
import onnxruntime
from onnx import shape_inference
import netron
import json

"""
Convert Pytorch model to Onnx
"""
model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)
torch.onnx.export(model, inputs, "onnx_model.onnx")

"""
Append new attributes to a specific operator
"""
onnx_model = onnx.load("onnx_model.onnx")
onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.save(onnx_model, "onnx_model.onnx")
graph = onnx_model.graph
node = graph.node

for n in node:  # n is NodeProto class
    if n.op_type == "Conv":
        h = helper.make_attribute("forward_time", [random.random()], "forward time consumed")
        n.attribute.append(h)

"""
Runtime
"""
options = onnxruntime.SessionOptions()
options.enable_profiling = True
ort_session = onnxruntime.InferenceSession("onnx_model.onnx", options)
ort_inputs = {ort_session.get_inputs()[0].name: inputs.detach().cpu().numpy()}
ort_outs = ort_session.run(None, ort_inputs)
prof_file = ort_session.end_profiling()
print(prof_file)

"""
Print onnx model attributes
"""
# for n in node:
#     if n.op_type == "Conv":
#         for one_attr in n.attribute:  # n.attribute is a list, one_attr is a AttributeProto class
#             print(one_attr.name, one_attr.ints, one_attr.floats)

"""
Display onnx model 
"""
with open(prof_file, "r") as f:
    sess_time = json.load(f)

"""
Display onnx model
"""
netron.start("onnx_model.onnx")
pass