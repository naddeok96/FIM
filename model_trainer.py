'''
This script will train a model and save it
'''
# Imports
import torch
from rand_lenet import RandLeNet
from first_layer_unitary_lenet import FstLayUniLeNet
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
net = FstLayUniLeNet(set_name = set_name,
                        gpu = gpu,
                        num_kernels_layer3 = 100)

# Load data
data = Data(gpu = gpu, set_name = "MNIST")

# Enter student network and curriculum data into an academy
academy  = Academy(net, data, gpu)

# Fit Model
academy.train(n_epochs = n_epochs)

# Calculate accuracy on test set
accuracy  = academy.test()
print(accuracy)

# Save Model
if save_model:
    # Define File Names
    filename  = "mnist_fstlay_unilenet_w_acc_" + str(int(round(rand_accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)
