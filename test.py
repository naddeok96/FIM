import torch
from copy import copy

batch_size  = 2
channel_num = 3
image_size  = 3

a = torch.eye(image_size).repeat(batch_size, channel_num, 1, 1)
u = torch.nn.init.orthogonal_(torch.empty(image_size, image_size)).repeat(batch_size, channel_num, 1, 1)


eig_values, eig_vectors = torch.symeig(tensor, eigenvectors = True, upper = True) 