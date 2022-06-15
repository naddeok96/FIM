# Imports
from os import uname
import io
from unitary_data_setup import UnitaryData
import torch
from models.classes.first_layer_unitary_net import FstLayUniNet
import numpy as np


# Hypers
original_root   = '../../../data/pytorch/MNIST/processed/test.pt'
unitary_root = '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/'
gpu = False
gpu_number = "7"

if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


# original_images, labels = torch.load(original_root)
# # unitary_images, labels = torch.load('../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/training.pt')
# # print(unitary_images.size())
# unitary_images = torch.empty((  int(1e4),
#                                 1,
#                                 28, 
#                                 28))

# for i in range(original_images.size(0)):
#     UA = torch.load(unitary_root + 'UA{}.pt'.format(i))
#     unitary_images[i,:,:,:] = UA

# torch.save((unitary_images, labels), '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/testing.pt')
a = '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/'
print(a.split('/')[-2])