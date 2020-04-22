'''
This script will train a model and save it
'''
# Imports
import torch
from unitary_lenet import UniLeNet
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy
import copy

# Hyperparameters
gpu = True
save_model = True
n_epochs = 1000
set_name = "MNIST"
seed = 1

frozen_layers = ["conv1.weight", "conv1.bias",
                "conv2.weight", "conv2.bias",
                "conv3.weight", "conv3.bias"]

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize
torch.manual_seed(seed)
uninet = UniLeNet(set_name = set_name,
                gpu = gpu,
                num_kernels_layer3 = 100)

torch.manual_seed(seed)
lenet = AdjLeNet(set_name = set_name,
                num_kernels_layer3 = 100)

# Load pretraind MNIST network
pre_trained_net = UniLeNet(set_name = set_name,
                    gpu = gpu,
                    num_kernels_layer3 = 100)
pre_trained_net.load_state_dict(torch.load('mnist_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
pre_trained_net.eval()

# Replace Random Feature Maps with Pretrained
uninet.conv1.weight.data = pre_trained_net.conv1.weight.data
uninet.conv1.bias.data   = pre_trained_net.conv1.bias.data
uninet.conv2.weight.data = pre_trained_net.conv2.weight.data
uninet.conv2.bias.data   = pre_trained_net.conv2.bias.data
uninet.conv3.weight.data = pre_trained_net.conv3.weight.data
uninet.conv3.bias.data   = pre_trained_net.conv3.bias.data

lenet.conv1.weight.data = pre_trained_net.conv1.weight.data
lenet.conv1.bias.data   = pre_trained_net.conv1.bias.data
lenet.conv2.weight.data = pre_trained_net.conv2.weight.data
lenet.conv2.bias.data   = pre_trained_net.conv2.bias.data
lenet.conv3.weight.data = pre_trained_net.conv3.weight.data
lenet.conv3.bias.data   = pre_trained_net.conv3.bias.data

# Load data
data = Data(gpu, set_name = "MNIST")

# Enter student network and curriculum data into an academy
uni_academy = Academy(uninet, data, gpu)
le_academy  = Academy(lenet, data, gpu)

# Fit Model
uni_academy.train(n_epochs = n_epochs,
              frozen_layers = frozen_layers)
le_academy.train(n_epochs = n_epochs,
                frozen_layers = frozen_layers)

# Calculate accuracy on test set
uni_accuracy = uni_academy.test()
le_accuracy  = le_academy.test()

# Save Model
if save_model:
    # Define File Names
    uni_filename = "mnist_unilenet_w_pretrained_kernels_w_acc_" + str(int(round(uni_accuracy * 100, 3))) + ".pt"
    le_filename  = "mnist_lenet_w_pretrained_kernels_w_acc_" + str(int(round(le_accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(uni_academy.net.state_dict(), uni_filename)
    torch.save(le_academy.net.state_dict(), le_filename)
