'''
This script will train a model and save it
'''
# Imports
import torch
import pickle
from data_setup import Data
from academy import Academy
from torchsummary import summary
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet
from models.classes.first_layer_unitary_effnet  import FstLayUniEffNet


# Hyperparameters
gpu         = False
save_model  = True
n_epochs    = 10000
set_name    = "CIFAR10"
model_name  = 'efficientnet-l2'
pretrained  = False
use_SAM     = True
seed        = 3

print("Epochs: ", n_epochs)
print("Pretrained: ", pretrained)
print("SAM: ", use_SAM)

# Push to GPU if necessary
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Declare seed and initalize network
torch.manual_seed(seed)

# Load data
# data = Data(gpu = gpu, set_name = set_name)
print(set_name, "Data Loaded")

# Unet
net = FstLayUniEffNet(  set_name = set_name,
                        gpu = gpu,
                        model_name = model_name,
                        pretrained = pretrained)
# net = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)

# net = FstLayUniLeNet(set_name = set_name, gpu = gpu)
# net.set_orthogonal_matrix()
# net.set_random_matrix()
# with open("models/pretrained/MNIST/weak_U.pkl", 'rb') as input:
#     net.U = pickle.load(input).type(torch.FloatTensor)
# print("U size: ", net.U.size())
net.eval()
summary(net, (3, 32, 32))
exit()
print(model_name, "Network Loaded")


# Enter student network and curriculum data into an academy
academy  = Academy(net, data, gpu, use_SAM)
print("Academy Set Up")

# Fit Model
print("Training")
academy.train(n_epochs = n_epochs,
              batch_size = 256)

# Calculate accuracy on test set
print("Testing")
accuracy  = academy.test()
print("Accuarcy: ", accuracy)

# Save Model
if save_model:
    # Define File Names
    filename  = model_name + "_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(academy.net.state_dict(), "models/pretrained/" + set_name + "2/" + filename)

    # Save U
    if net.U is not None:
        torch.save(net.U, "models/pretrained/" + set_name + "2/U_" + filename)
