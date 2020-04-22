'''
This script will train a model and save it
'''
# Imports
import torch
from unitary_lenet import UniLeNet
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy

# Hyperparameters
gpu = True
save_model = True
n_epochs = 1
set_name = "MNIST"
seed = 1

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(seed)
unilenet = UniLeNet(set_name = set_name,
                    gpu = gpu,
                    num_kernels_layer3 = 100)

# Load data
data = Data(gpu, set_name = "MNIST")

# Enter student network and curriculum data into an academy
uni_academy  = Academy(unilenet, data, gpu)

# Fit Model
uni_academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
uni_accuracy  = uni_academy.test()

# Save Model
if save_model:
    # Define File Names
    uni_filename  = "mnist_unilenet_w_acc_" + str(int(round(uni_accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(uni_academy.net.state_dict(), uni_filename)
