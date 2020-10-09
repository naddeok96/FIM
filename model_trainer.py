'''
This script will train a model and save it
'''
# Imports
import torch
from data_setup import Data
from academy import Academy
from models.classes.rand_lenet                  import RandLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet
from models.classes.first_layer_rand_lenet      import FstLayRandLeNet
from models.classes.unitary_lenet               import UniLeNet
from models.classes.adjustable_lenet            import AdjLeNet

# Hyperparameters
gpu         = False
save_model  = True
n_epochs    = 1000
set_name    = "MNIST"
seed        = 100

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Declare seed and initalize network
torch.manual_seed(seed)
net = AdjLeNet(set_name = set_name)

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
    filename  = "seed100_lenet_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), filename)

    # # Save U
    # if net.U is not None:
    #     torch.save(net.U, "U_" + filename)
