'''
This code will be used as the main code to run all classes
'''

# Imports
from Adjustable_LeNet import AdjLeNet
from MNIST_Setup import MNIST_Data
from Gym import Gym
from One_Step_Spectral_Attack import OSSA
from PIL import Image
import torchvision.transforms.functional as F
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Initialize
net = AdjLeNet(num_classes = 10,
               num_kernels_layer1 = 6, 
               num_kernels_layer2 = 16, 
               num_kernels_layer3 = 120,
               num_nodes_fc_layer = 84)

data = MNIST_Data()

detministic_model = Gym(net = net,
                        data = data)

# Fit Model
print(data.get_single_image())




