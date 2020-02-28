'''
This script will train a model and save it
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from cifar10_setup import CIFAR10_Data
from academy import Academy

# Hyperparameters
gpu = True
n_epochs = 100

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize
net = AdjLeNet(CIFAR10 = True)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

data = CIFAR10_Data(gpu)
academy = Academy(net, data, gpu)

# Fit Model
academy.train(n_epochs = n_epochs)
accuracy = academy.test()

# Save Model
filename = "cifar10_lenet_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
torch.save(academy.net.state_dict(), filename)
