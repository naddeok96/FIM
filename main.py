'''
This code will be used as the main code to run all classes
'''

# Imports
import torch
from Adjustable_LeNet import AdjLeNet
from MNIST_Setup import MNIST_Data
from Gym import Gym
from One_Step_Spectral_Attack import OSSA
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import operator

'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
'''

# Initialize
net = AdjLeNet(num_classes = 10,
               num_kernels_layer1 = 6, 
               num_kernels_layer2 = 16, 
               num_kernels_layer3 = 120,
               num_nodes_fc_layer = 84)

data = MNIST_Data()

detministic_model = Gym(net = net, data = data)

criterion = torch.nn.CrossEntropyLoss()

# Fit Model
accuracy = detministic_model.train(n_epochs = 0)

image, label, show_image = data.get_single_image()

attack = OSSA(net, image, label)

