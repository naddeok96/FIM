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
import matplotlib.pyplot as plt
import operator

# Hyperparameters
gpu = True
plot = False
n_epochs = 10

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize
net = AdjLeNet(num_classes = 10,
               num_kernels_layer1 = 6, 
               num_kernels_layer2 = 16, 
               num_kernels_layer3 = 120,
               num_nodes_fc_layer = 84)
data = MNIST_Data()
detministic_model = Gym(net, data, gpu)

# Fit Model
accuracy = detministic_model.train(n_epochs = n_epochs)

# Generate an Attack using OSSA
image, label, show_image = data.get_single_image()

attack = OSSA(net, image, label, gpu = gpu)

# Test Attack
prediction = detministic_model.get_single_prediction(image)
attack_prediction = detministic_model.get_single_prediction(image - attack.attack_perturbation)



# Display
print("--------------------------------------")
print("Deterministic Model Accuracy: ", accuracy)
print("Image Label: ", label.item())
print("Deterministic Model Prediction: ", prediction.item())
print("Deterministic Model Attack Prediction: ", attack_prediction.item())
print("--------------------------------------")

if plot == True:
    rows = 1
    cols = 3
    figsize = [8, 4]
    fig= plt.figure(figsize=figsize)
    fig.suptitle('OSSA Attack Summary', fontsize=16)

    ax1 = fig.add_subplot(rows, cols, 1)
    ax1.imshow(show_image)
    ax1.set_title("Orginal Image")

    ax2 = fig.add_subplot(rows, cols, 2)
    ax2.imshow(attack.attack_perturbation)
    ax2.set_title("Attack Perturbation")

    ax3 = fig.add_subplot(rows, cols, 3)
    ax3.imshow(show_image + attack.attack_perturbation)
    ax3.set_title("Attack")

    plt.show()
