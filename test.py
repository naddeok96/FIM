import torch
from pprint import pprint
# pprint(torch.hub.list("chenyaofo/pytorch-cifar-models", force_reload=True))
# exit()

net = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)