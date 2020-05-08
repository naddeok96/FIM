'''
This script will train a model and save it
'''
# Imports
import torch
from rand_lenet import RandLeNet
from unitary_lenet import UniLeNet
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy

# Hyperparameters
gpu = False
save_model = False
n_epochs = 0
set_name = "MNIST"
seed = 1

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(seed)
randlenet = RandLeNet(set_name = set_name,
                        gpu = gpu,
                        num_kernels_layer3 = 100)

# Load data
data = Data(gpu = gpu, set_name = "MNIST")

# Enter student network and curriculum data into an academy
rand_academy  = Academy(randlenet, data, gpu)

# Fit Model
rand_academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
rand_accuracy  = rand_academy.test()

# Save Model
if save_model:
    # Define File Names
    rand_filename  = "mnist_randlenet_w_acc_" + str(int(round(rand_accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(rand_academy.net.state_dict(), rand_filename)
