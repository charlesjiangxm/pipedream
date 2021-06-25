import torch
import torchvision
import graphviz
import torchprof

# construct model
model = torchvision.models.resnet18().cpu()
x = torch.rand([1, 3, 224, 224]).cpu()
label = torch.ones([1]).long().cpu()

# profile and get the result
with torchprof.Profile(model, use_cuda=False, profile_memory=False) as prof:
    with torch.no_grad():
        model(x)

# construct node
# dot = graphviz.Digraph(comment='resnet18')
# for event_name, event_list in prof.trace_profile_events.items():
#     for event in event_list:
#         for node in event:
#             dot.node(str(node.id), str(node.id) + " " + node.name + " " + node.cpu_time_total_str)
#
# # construct edge
# for event_name, event_list in prof.trace_profile_events.items():
#     for event in event_list:
#         for node in event:
#             for child_node in node.cpu_children:
#                 dot.edge(str(node.id), str(child_node.id))
#
# # render
# dot.render('asdf.dot', view=True)

print(prof.display(show_events=False))

pass
