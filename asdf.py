# import torch
# import torch.nn as nn
# from torchviz import make_dot
# import torchvision
#
# model_name = "rn18"
# model = torchvision.models.resnet18()
# example = torch.rand(1, 3, 224, 224)
# torch_script_model = torch.jit.trace(model, example)
# with torch.no_grad():
#     y = torch_script_model.forward(example)
#     asdf = make_dot(y)
#     asdf.render('asdf.dot', view=True)
#     pass