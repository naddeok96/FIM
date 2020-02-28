'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from mnist_setup import MNIST_Data
from academy import Academy
from information_geometry import InfoGeo
import torchvision.transforms.functional as F
import operator
import numpy as np

# Hyperparameters
gpu = False
plot = False
save_set = False

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize data
data = MNIST_Data(gpu)

# Load pretraind network
net = AdjLeNet(num_classes = 10,
                             num_kernels_layer1 = 6, 
                             num_kernels_layer2 = 16, 
                             num_kernels_layer3 = 120,
                             num_nodes_fc_layer = 84)
net.load_state_dict(torch.load('trained_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
net.eval()

# Get a single image
image, label = data.get_single_image(index = 0)

# Initialize InfoGeo object
info_geo = InfoGeo(net, image, label, EPSILON = 50)

# Calculate FIM
info_geo.get_FIM()
info_geo.get_attack()
info_geo.get_prediction()

# Display
info_geo.plot_attack()



'''
if save_set == True:
    f = open("adverserial_image_set.txt","w")
    f.write( str(adverserial_image_set) )
    f.close()
    f = open("label_set.txt","w")
    f.write( str(label_set) )
    f.close()
'''



