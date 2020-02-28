'''
This script will train a model and save it
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy

# Hyperparameters
gpu = True
n_epochs = 200

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize
net = AdjLeNet(set_name = "CIFAR10",
               num_kernels_layer1 = 12, 
               num_kernels_layer2 = 32, 
               num_kernels_layer3 = 240,
               num_nodes_fc_layer = 168)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

data = Data(gpu, set_name = "CIFAR10")
academy = Academy(net, data, gpu)

# Fit Model
academy.train(n_epochs = n_epochs)
accuracy = academy.test()

# Save Model
filename = "cifar10_lenet_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
torch.save(academy.net.state_dict(), filename)
