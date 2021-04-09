import torch
from copy import copy

batch_size  = 2
channel_num = 3
image_size  = 3

a = torch.eye(image_size).repeat(batch_size, channel_num, 1, 1)
u = torch.nn.init.orthogonal_(torch.empty(image_size, image_size))

U = copy(u.view(1, image_size, image_size)).repeat(batch_size * channel_num, 1, 1)

r = torch.bmm(U, a.view(batch_size * channel_num, image_size, image_size)).view(batch_size, channel_num, image_size, image_size)

print("A")
print(a)
print(a.size())

print("u")
print(u)
print(u.size())

print("U")
print(U)
print(U.size())

print("R")
print(r)
print(r.size())