# Imports
from os import uname
import io
from unitary_data_setup import UnitaryData
import torch
from models.classes.first_layer_unitary_net import FstLayUniNet
import numpy as np
from data_setup import Data


# Hypers
original_root   = '../../../data/pytorch/MNIST/processed/test.pt'
unitary_root = '../../../data/naddeok/optimal_UA_for_lenet_w_acc_98/train/'

gpu = False
gpu_number = "2"

if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


original_images, labels = torch.load(original_root)
data = Data(gpu          = gpu, 
            set_name     = "MNIST", 
            test_batch_size = int(1e4))
import torchvision.datasets as datasets
test_set = datasets.MNIST(root='../../../data/pytorch/',
                                            train = False,
                                            download = True) 

test_loader = torch.utils.data.DataLoader(test_set,
                                            batch_size = int(1e4),
                                            shuffle = False)
print(test_loader)
next(iter(test_loader))
batch_images, _ = next(iter(data.test_loader))
# print(torch.max(tl_images))
# print(torch.min(tl_images))
print(torch.max(batch_images))
print(torch.min(batch_images))
print(torch.max(original_images))
print(torch.min(original_images))
exit()
# unitary_images, labels = torch.load('../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/training.pt')
# print(unitary_images.size())
unitary_images = torch.empty((  int(1e4),
                                1,
                                28, 
                                28))

for i in range(int(1e4)):
    UA = torch.load(unitary_root + 'UA{}.pt'.format(i))
    unitary_images[i,:,:,:] = UA

torch.save((unitary_images, labels), '../../../data/naddeok/optimal_UA_for_lenet_w_acc_98/train/training.pt')
# torch.save((unitary_images, labels), '../../../data/naddeok/optimal_UA_for_lenet_w_acc_98/test/testing.pt')
