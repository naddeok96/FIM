# Imports
from models.classes.first_layer_unitary_lenet import FstLayUniLeNet
from prettytable import PrettyTable
from data_setup import Data
from academy import Academy
import numpy as np
import torch

# Hyperparameters
gpu         = True
save_model  = True
n_epochs    = 1
set_name    = "MNIST"
seed        = 100
reg_train_ratio = 0.5

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Declare seed and initalize network
net = FstLayUniLeNet(set_name = set_name,
                    gpu = gpu,
                    seed = seed,
                    U = None)
U = net.get_orthogonal_matrix(784)


# Load data
data = Data(gpu = gpu, set_name = "MNIST")

# Enter student network and curriculum data into an academy
academy  = Academy(net, data, gpu)

# Swich between U and Reg
count = 0
for _ in range(n_epochs):
    count += 1
    print(count)

    # Determine Unitary or not
    if reg_train_ratio < np.random.uniform():
        academy.net.U = None
    else:
        academy.net.U = U

    # Fit Model
    academy.train(n_epochs = 1)

# Calculate accuracy on test set
academy.net.U = None
reg_accuracy  = academy.test()
print("Regular Network Accuracy: ", reg_accuracy)

academy.net.U = U
U_accuracy  = academy.test()
print("Unitary Network Accuracy: ", U_accuracy)
accuracy = (reg_accuracy + U_accuracy)/2
print("Average Network Accuracy: ", accuracy)

# Save Model
if save_model:
    # Define File Names
    filename  = "mixed_U_net_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)

    # Save U
    if net.U is not None:
        torch.save(net.U, "U_for_" + filename)