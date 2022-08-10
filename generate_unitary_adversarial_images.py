# Imports
from logging import root
import os
import torch
from data_setup import Data

# Hypers
set_name   = 'MNIST'
gpu        = True
gpu_number = "3"
attack_types = ["FGSM"]
epsilons     =  torch.linspace(0,255,52, dtype=torch.uint8)

unitary_root    = "../../../data/naddeok/mnist_U_files/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test"
pert_root       = unitary_root + "/adversarial_perturbations"
attack_root     = unitary_root + "/adversarial_attacks"

# Load Data
data = Data(set_name = set_name,
            test_batch_size = 1,
            gpu = gpu,
            maxmin=True)
            
# Ensure save folder exists
net_name = pert_root.split("/")[-1]
print(net_name)
exit()
for attack in attack_types:
    # Pert
    if not os.path.isdir(pert_root):
        os.mkdir(pert_root)

    # Network
    if not os.path.isdir(attack_root):
        os.mkdir(attack_root)

    # Attacks
    for attack in attack_types:
        if not os.path.isdir(attack_root + "/" + attack + "/"):
            os.mkdir(attack_root + "/" + attack + "/")

        # Epsilons
        for epsilon in epsilons:
            if not os.path.isdir(attack_root + "/" + attack + "/LInf{}/".format(epsilon)):
                os.mkdir(attack_root + "/" + attack + "/LInf{}/".format(epsilon))

# Apply Perturbations
for attack in attack_types:
    for i, (image, labels) in enumerate(data.test_loader):
        
        pert = torch.load(pert_root + attack + "/P{}.pt".format(i))

        if gpu:
            image = image.cuda()
            pert  = pert.cuda()
        
        assert set_name == "MNIST", "Rescale to 255 is only supported for single channel"
        image  = 255 * ((image * data.std[0]) + data.mean[0]).view(1,1,-1)
        pert   = 255 * ((pert   * data.std[0]) + data.mean[0])


        for epsilon in epsilons:

            adv_image = torch.clamp(image + (epsilon/torch.max(pert)) * pert, 0, 255)

            if epsilon > 5:
                print(epsilon)
                print(torch.max((epsilon/torch.max(pert)) * pert))
                print(torch.min(adv_image))
                from torchvision.utils import save_image
                save_image(adv_image.view(1,28,28), "TEMP.png")
                exit()
        
# # Join all images into one dataset
# unitary_images = torch.empty((  int(1e4),
#                                 1,
#                                 28, 
#                                 28))

# for i in range(original_images.size(0)):
#     UA = torch.load(unitary_root + 'UA{}.pt'.format(i))
#     unitary_images[i,:,:,:] = UA

# torch.save((unitary_images, labels), '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/testing.pt')