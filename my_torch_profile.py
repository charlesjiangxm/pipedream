import torch
import torchvision
import graphviz
import torch.profiler as profiler
# import torchprof

# construct model
model = torchvision.models.resnet18().cuda()
x = torch.rand([1, 3, 224, 224]).cuda()
label = torch.ones([1]).long().cuda()

# profile and get the result
with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU,
                    profiler.ProfilerActivity.CUDA],
        record_shapes=False) as prof:
    with torch.no_grad():
        model(x)

# construct node
dot = graphviz.Digraph(comment='vgg11')
for i, node in enumerate(prof.profiler.function_events):
    dot.node(str(node.id), str(node.id) + " " + node.name + " " + node.cuda_time_total_str)

# construct edge
for node in prof.profiler.function_events:
    for child_node in node.cpu_children:
        dot.edge(str(node.id), str(child_node.id))

# render
dot.render('asdf.dot', view=True)

pass
