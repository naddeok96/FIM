
import matplotlib
from torch.utils import data
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import torchvision.transforms as transforms
from copy import copy

transform = transforms.Compose([transforms.ToTensor()])

data_set = datasets.ImageFolder(root      = '../../../data/tiny-imagenet-200/train/',
                                transform = transform)

data_loader = torch.utils.data.DataLoader(  data_set,
                                            batch_size = 500,
                                            shuffle = True)

def add_stats(mean1, std1, weight1, mean2, std2, weight2):
    '''
    Takes stats of two sets (assumed to be from the same distribution) and combines them
    Method from https://www.statstodo.com/CombineMeansSDs_Pgm.php
    '''
    # Calculate E[x] and E[x^2] of each
    sig_x1 = weight1 * mean1
    sig_x2 = weight2 * mean2

    sig_xx1 = ((std1 ** 2) * (weight1 - 1)) + (((sig_x1 ** 2) / weight1))
    sig_xx2 = ((std2 ** 2) * (weight2 - 1)) + (((sig_x2 ** 2) / weight2))

    # Calculate sums
    tn  = weight1 + weight2
    tx  = sig_x1  + sig_x2
    txx = sig_xx1 + sig_xx2

    # Calculate combined stats
    mean = tx / tn
    std = np.sqrt((txx - (tx**2)/tn) / (tn - 1))

    return mean, std, tn

for i, (inputs, labels) in enumerate(data_loader):
    print(i, " of ", len(data_loader))
    
    batch_size = inputs.size(0)
    m0b = torch.mean(inputs[:, 0, :, :])
    m1b = torch.mean(inputs[:, 1, :, :])
    m2b = torch.mean(inputs[:, 2, :, :])

    s0b = torch.std(inputs[:, 0, :, :])
    s1b = torch.std(inputs[:, 1, :, :])
    s2b = torch.std(inputs[:, 2, :, :])

    if i == 0:
        m0 = copy(m0b)
        m1 = copy(m1b)
        m2 = copy(m2b)

        s0 = copy(s0b)
        s1 = copy(s1b)
        s2 = copy(s2b)
        total_size = copy(batch_size)

    else:
        m0, s0, _          = add_stats(m0b, s0b, batch_size, m0, s0, total_size)
        m1, s1, _          = add_stats(m1b, s1b, batch_size, m1, s1, total_size)
        m2, s2, total_size = add_stats(m2b, s2b, batch_size, m2, s2, total_size)

print("Total tested:", total_size)
print("Channel 0", m0, s0)
print("Channel 1", m1, s1)
print("Channel 2", m2, s2)


# img = torchvision.utils.make_grid((inputs[10]))
# plt.imshow(np.transpose(img.cpu().numpy(), (1 , 2 , 0)))
# plt.savefig("AAAAAAAAAAAAAAAAAAAAAAAAAAAAH.png")