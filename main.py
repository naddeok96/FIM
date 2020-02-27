'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from mnist_setup import MNIST_Data
from gym import Gym
from one_step_spectral_attack import OSSA
import torchvision.transforms.functional as F
import operator

# Hyperparameters
gpu = False
plot = False
save_set = False

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize
data = MNIST_Data()
detministic_model = torch.load('trained_lenet_w_acc_98.pt', map_location=torch.device('cpu'))
gym = Gym(detministic_model, data, gpu = False)

# Test model
accuracy = gym.test()

'''
if save_set == True:
    f = open("adverserial_image_set.txt","w")
    f.write( str(adverserial_image_set) )
    f.close()
    f = open("label_set.txt","w")
    f.write( str(label_set) )
    f.close()
'''



