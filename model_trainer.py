'''
This script will train a model and save it
'''
# Imports
import torch
import pickle
from data_setup import Data
from academy import Academy
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet

# Hyperparameters
gpu         = True
save_model  = True
n_epochs    = 100
set_name    = "MNIST"
model_name  = "RNet"
seed        = 100

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Declare seed and initalize network
torch.manual_seed(seed)


# Load data
data = Data(gpu = gpu, set_name = set_name)
print(set_name, " Data Loaded")

# Unet
# net = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)

net = FstLayUniLeNet(set_name = set_name, gpu = gpu)
# net.set_orthogonal_matrix()
net.set_random_matrix()
# with open("models/pretrained/MNIST/weak_U.pkl", 'rb') as input:
#     net.U = pickle.load(input).type(torch.FloatTensor)
print("U size: ", net.U.size())
net.eval()
print(model_name, " Network Loaded")


# Enter student network and curriculum data into an academy
academy  = Academy(net, data, gpu)
print("Academy Set Up")

# Fit Model
print("Training")
academy.train(n_epochs = n_epochs,
              batch_size = 256)

# Calculate accuracy on test set
accuracy  = academy.test()
print("Testing Done")
print(accuracy)

# Save Model
if save_model:
    # Define File Names
    filename  = model_name + "_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), "models/pretrained/" + set_name + "/" + filename)

    # Save U
    if net.U is not None:
        torch.save(net.U, "models/pretrained/" + set_name + "/U_" + filename)
