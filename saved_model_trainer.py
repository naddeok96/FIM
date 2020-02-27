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

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

data = MNIST_Data()
gym = Gym(net, data, gpu)

# Fit Model
gym.train(n_epochs = n_epochs)
accuracy = gym.test()

# Save Model
filename = "trained_lenet_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
torch.save(gym.net.state_dict(), filename)
