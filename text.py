import numpy as np
import torch
import torchvision.transforms as transforms

N = 4 
a = np.random.random_integers(-2000,2000,size=(N,N))
a = (a + a.T)/2
b = np.random.random_integers(-2000,2000,size=(N,N))
b = (b + b.T)/2

matrices = [a, b]
c1 = torch.stack([transforms.ToTensor()(matrix) for matrix in matrices])

N = 4 
a = np.random.random_integers(-2000,2000,size=(N,N))
a = (a + a.T)/2
b = np.random.random_integers(-2000,2000,size=(N,N))
b = (b + b.T)/2

matrices = [a, b]
c2 = torch.stack([transforms.ToTensor()(matrix) for matrix in matrices])

print(c1)
print(c2)
print(torch.cat((c1, c2), 0))

