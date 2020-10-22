# Imports
from models.classes.first_layer_unitary_dense_net import FstLayUniDenseNet
from prettytable import PrettyTable, ALL
from data_setup import Data
from academy import Academy
import numpy as np
import torch

# Hyperparameters
gpu         = True
save_model  = True
n_epochs    = 1000
set_name    = "MNIST"
seed        = 100
reg_train_ratio = 0.5

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Declare seed and initalize network
net = FstLayUniDenseNet(set_name = set_name,
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
table = PrettyTable(hrules = ALL)
table.field_names =(["Epoch", "Unitary Test Accuracy", "Regular Test Accuracy", "Test Accuracy"])
for _ in range(n_epochs):
    count += 1
    if count % 100 == 0:
        # Calculate accuracy on test set
        academy.net.U = U
        U_test = round(academy.test() * 100, 2)

        academy.net.U = None
        reg_test = round(academy.test() * 100, 2)

        accuracy = round((U_test + reg_test)/2, 2)

        table.add_row([count, U_test, reg_test, accuracy])
        print(table)

    # Determine Unitary or not
    if reg_train_ratio < np.random.uniform():
        academy.net.U = None
    else:
        academy.net.U = U

    # Fit Model
    academy.train(n_epochs = 1)

# Save Model
if save_model:
    # Define File Names
    filename  = "mixed_U_dense_net_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)

    # Save U
    if net.U is not None:
        torch.save(net.U, "U_for_" + filename)