# Imports
from logging import root
import os
import torch

# Hypers
set_name   = 'MNIST'
gpu        = True
gpu_number = "1"
attack_types = ["CW2"]
epsilons     =  torch.linspace(0,255,52, dtype=torch.uint8)

pert_root       = "../../../data/naddeok/mnist_adversarial_perturbations/lenet_w_acc_97/"
unitary_root    = "../../../data/naddeok/optimal_U_for_lenet_w_acc_98/test/"
save_root       = "../../../data/naddeok/mnist_adversarial_attacks/"

# Ensure save folder exists
net_name = pert_root.split("/")[-1]
for attack in attack_types:
    # Parent
    if not os.path.isdir(save_root):
        os.mkdir(save_root)

    # Network
    if not os.path.isdir(save_root + net_name + "/"):
        os.mkdir(save_root + net_name + "/")

    # Attacks
    for attack in attack_types:
        if not os.path.isdir(save_root + net_name + "/" + attack + "/"):
            os.mkdir(save_root + net_name + "/" + attack + "/")

        # Epsilons
        for epsilon in epsilons:
            if not os.path.isdir(save_root + net_name + "/" + attack + "/LInf%d.pt".format(epsilon)):
                os.mkdir(save_root + net_name + "/" + attack + "/LInf{}/".format(epsilon))


for attack in attack_types:
    for epsilon in epsilons:
        
# Join all images into one dataset
unitary_images = torch.empty((  int(1e4),
                                1,
                                28, 
                                28))

for i in range(original_images.size(0)):
    UA = torch.load(unitary_root + 'UA{}.pt'.format(i))
    unitary_images[i,:,:,:] = UA

# torch.save((unitary_images, labels), '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/testing.pt')