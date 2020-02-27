'''
This script will train a model and save it
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from mnist_setup import MNIST_Data
from gym import Gym

# Hyperparameters
gpu = True
n_epochs = 100

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize
net = AdjLeNet(num_classes = 10,
               num_kernels_layer1 = 6, 
               num_kernels_layer2 = 16, 
               num_kernels_layer3 = 120,
               num_nodes_fc_layer = 84)
data = MNIST_Data()
detministic_model = Gym(net, data, gpu)

# Fit Model
accuracy = detministic_model.train(n_epochs = n_epochs)

# Save Model
filename = "trained_lenet_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
torch.save(detministic_model.state_dict(), filename)
