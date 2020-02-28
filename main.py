'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy
from information_geometry import InfoGeo
import torchvision.transforms.functional as F
import operator
import numpy as np

# Hyperparameters
gpu = False
set_name = "MNIST"
save_set = False

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize data
data = Data(gpu, set_name = set_name)

# Load pretraind network
net = AdjLeNet(set_name = set_name, 
                num_classes = 10,
                num_kernels_layer1 = 6, 
                num_kernels_layer2 = 16, 
                num_kernels_layer3 = 120,
                num_nodes_fc_layer = 84)
net.load_state_dict(torch.load('mnist_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
net.eval()

# Get a single image
image, label = data.get_single_image(index = 0)

# Initialize InfoGeo object
info_geo = InfoGeo(net, image, label, EPSILON = 20)

# Calculate FIM
info_geo.get_FIM() 

# Calculate
info_geo.get_attack()
info_geo.get_prediction()


# Display
info_geo.plot_attack()

# Save adverserial image set
if save_set == True:
    f = open("adverserial_image_set.txt","w")
    f.write( str(adverserial_image_set) )
    f.close()
    f = open("label_set.txt","w")
    f.write( str(label_set) )
    f.close()




