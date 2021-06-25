# import torch
# import torch.nn as nn
# import torchvision
#
#
# class MyModule(nn.Module):
#     def __init__(self):
#         super(MyModule, self).__init__()
#         self.conv1 = nn.Conv2d(1, 3, 3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x
#
#
# model = torchvision.models.resnet18()  # 实例化模型
# trace_module = torch.jit.trace(model, torch.rand(1, 3, 224, 224))
# print(trace_module.code)  # 查看模型结构
# output = trace_module(torch.ones(1, 3, 224, 224))  # 测试
# trace_module('model.pt')  # 模型保存


# import tensorwatch as tw
# import torchvision.models
#
#
# alexnet_model = torchvision.models.resnet18().cuda()
# # tw.draw_model(alexnet_model, [1, 3, 224, 224])
# out = tw.model_stats(alexnet_model, [1, 3, 224, 224])
# print(out)
#
# pass

import torch
import torchvision
from torch.profiler import profile, ProfilerActivity
import graphviz

# An instance of your model.
model = torchvision.models.alexnet().cuda()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    traced_script_module.forward(example)

# construct node
dot = graphviz.Digraph(comment='rn18')
for i, node in enumerate(prof.profiler.function_events):
    if node.cpu_parent is not None and node.cpu_parent.name == "forward":
        dot.node(str(node.id), str(node.id) + " " + node.name + " " + node.cuda_time_total_str)

# construct edge
# for node in prof.profiler.function_events:
#     # for child_node in node.cpu_children:
#     #     if node.name != "forward":
#     #         dot.edge(str(node.id), str(child_node.id))
#
#     if node.cpu_parent is not None and node.cpu_parent.name != "forward":
#         dot.edge(str(node.cpu_parent.id), str(node.id))

traced_script_module.save("rn18.pt")