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
gpu = False
plot = False

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize
data = MNIST_Data()
detministic_model = torch.load('trained_lenet_w_acc_98.pt', map_location=torch.device('cpu'))

# Generate an Attack using OSSA
total_tests = data.test_set.data.size()[0]
correct = 0
correct_adv = 0
for i in range(total_tests):
    image, label, show_image = data.get_single_image(index = i) 

    
    attack = OSSA(net = detministic_model.net, 
                  image = image, 
                  label = label,
                  gpu = gpu)

    adverserial_image = image  + attack.attack_perturbation

    output = detministic_model.net(image)
    _, predicted = torch.max(output.data, 1)
    correct += (predicted == label).item()
    
    output_adv = detministic_model.net(adverserial_image)
    _, predicted_adv = torch.max(output_adv.data, 1)
    correct_adv += (predicted_adv == label).item()

accuracy     = correct/total_tests
accuracy_adv = correct_adv/total_tests

# Display
print("--------------------------------------")
print("Deterministic Accuracy: ", accuracy)
print("Adverserial Accuracy: ", accuracy_adv)
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
