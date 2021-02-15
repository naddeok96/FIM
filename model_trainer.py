'''
This script will train a model and save it
'''
# Imports
import torch
import pickle
from data_setup import Data
from academy import Academy
from models.classes.rand_lenet                  import RandLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet
from models.classes.first_layer_rand_lenet      import FstLayRandLeNet
from models.classes.unitary_lenet               import UniLeNet
from models.classes.adjustable_lenet            import AdjLeNet

# Hyperparameters
gpu         = True
save_model  = False
n_epochs    = 1000
set_name    = "ImageNet"
seed        = 200

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Declare seed and initalize network
torch.manual_seed(seed)
# Unet
# net = FstLayUniLeNet(set_name = set_name, gpu = gpu)
# with open("models/pretrained/high_R_U.pkl", 'rb') as input:
#     net.U = pickle.load(input).type(torch.FloatTensor)
net = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
net.eval()
print("Network Loaded")

# Load data
data = Data(gpu = gpu, set_name = set_name, test_batch_size = 128)
print("Data Loaded")

# Enter student network and curriculum data into an academy
academy  = Academy(net, data, gpu)
print("Academy Set Up")

# Fit Model
# academy.train(n_epochs = n_epochs,
#               batch_size = 256)

# Calculate accuracy on test set
accuracy  = academy.test()
print("Testing Done")
print(accuracy)

# Save Model
if save_model:
    # Define File Names
    filename  = "CIFAR10_200seed_lenet_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), "models/pretrained/" + filename)

    # Save U
    if net.U is not None:
        torch.save(net.U, "models/pretrained/U_" + filename)
