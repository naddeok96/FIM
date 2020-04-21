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
n_epochs = 1000
set_name = "MNIST"
seed = 1

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.manual_seed(seed)
lenet = AdjLeNet(set_name = set_name,
                num_kernels_layer3 = 100)

# Load data
data = Data(gpu, set_name = "MNIST")

# Enter student network and curriculum data into an academy
le_academy  = Academy(lenet, data, gpu)

# Fit Model
le_academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
le_accuracy  = le_academy.test()

# Save Model
if save_model:
    # Define File Names
    le_filename  = "mnist_lenet_w_acc_" + str(int(round(le_accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(le_academy.net.state_dict(), le_filename)
