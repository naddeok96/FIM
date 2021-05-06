import torch
from copy import copy
from efficientnet_pytorch import EfficientNet
from torchsummary import summary

net = EfficientNet.from_pretrained(model_name = 'efficientnet-b8',
                                                num_classes = 10,
                                                image_size  = 32)
# print(net.summary)
# summary(net, (3, 32, 32))