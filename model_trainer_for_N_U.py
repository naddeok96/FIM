'''
This script will train a model and save it
'''
# Imports
import torch
import pickle5 as pickle
from data_setup import Data
from academy import Academy
import matplotlib.pyplot as plt
from models.classes.rand_lenet                  import RandLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet
from models.classes.first_layer_rand_lenet      import FstLayRandLeNet
from models.classes.unitary_lenet               import UniLeNet
from models.classes.adjustable_lenet            import AdjLeNet

# Hyperparameters
gpu         = True
save_model  = True
n_epochs    = 1000
set_name    = "MNIST"
seed        = 100

# Push to GPU if necessary
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Declare seed and initalize network
torch.manual_seed(seed)

# Load U's
N = 10
with open("models/pretrained/" + str(N) + "_Rs" + '.pkl', 'rb') as input:
    Rs = pickle.load(input)

# Load data
data = Data(gpu = gpu, set_name = "MNIST")

accuracies = []
# Train
for i in range(len(Rs)):
    # Reset Unet
    net = FstLayUniLeNet(set_name = set_name, gpu = gpu)

    with open("models/pretrained/" + str(i)+ "_of_" + str(N) + "_Us" + '.pkl', 'rb') as input:
        net.U = pickle.load(input).type(torch.FloatTensor)

    print("Working on ", i + 1, " of ", len(Rs))
    # Enter student network and curriculum data into an academy
    academy  = Academy(net, data, gpu)

    # Fit Model
    academy.train(n_epochs = n_epochs,
                  batch_size = 124)

    # Calculate accuracy on test set
    accuracies.append(academy.test())
    print("Accuracy for Rotation " + str(i) + ": " + str(accuracies[i]))

    # Save Model
    if save_model:
        # Define File Names
        filename  = "models/pretrained/U" + str(i + 1) + "_of_" + str(N) + "_net_w_acc_" + str(int(round(accuracies[i] * 100, 3))) + ".pt"
        
        # Save Models
        torch.save(academy.net.state_dict(), filename)

        # Save U
        # if net.U is not None:
        #     torch.save(net.U, "U_" + filename)

plt.plot(Rs, accuracies)
plt.xlabel("Roations")
plt.ylabel("Accuracy")
plt.show()