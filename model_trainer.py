'''
This script will train a model and save it
'''
# Imports
import torch
from unitary_lenet import UniLeNet
from data_setup import Data
from academy import Academy

# Hyperparameters
gpu = True
save_model = False

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize
net = UniLeNet(set_name = "MNIST",
                num_kernels_layer3 = 100)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in net.state_dict():
    print(param_tensor, "\t", net.state_dict()[param_tensor].size())

# Load data
data = Data(gpu, set_name = "MNIST")

# Enter student network and curriculum data into an academy
academy = Academy(net, data, gpu)

# Fit Model
academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
accuracy = academy.test()
print(accuracy)

# Save Model
if save_model:
    filename = "mnist_unilenet_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    torch.save(academy.net.state_dict(), filename)
