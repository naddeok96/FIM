# Imports
from logging import root
import os
import torch
from data_setup import Data

# Hypers
set_name   = 'MNIST'
gpu        = True
gpu_number = "2"
attack_types = ["FGSM"]
epsilons     =  torch.linspace(0,255,52, dtype=torch.uint8)

unitary_root    = "../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test"
pert_root       = unitary_root + "/adversarial_perturbations"
attack_root     = unitary_root + "/adversarial_attacks"

# Decalre machines
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Load Data
data = Data(set_name = set_name,
            test_batch_size = 1,
            gpu = gpu,
            maxmin=True)
            
# Ensure save folder exists
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
labels = torch.zeros(int(1e4))
for attack in attack_types:
    for i, (image, label) in enumerate(data.test_loader):
        labels[i] = label
        pert = torch.load(pert_root + "/" + attack + "/P{}.pt".format(i))

        if gpu:
            image = image.cuda()
            pert  = pert.cuda()
        
        assert set_name == "MNIST", "Rescale to 255 is only supported for single channel"
        image  = 255 * ((image * data.std[0]) + data.mean[0]).view(1,1,-1)
        pert   = 255 * ((pert  * data.std[0]) + data.mean[0])


        for epsilon in epsilons:
            scaled_pert = ((epsilon/torch.max(pert)) * pert)
            adv_image = torch.clamp(image + scaled_pert, 0, 255)

            torch.save(adv_image, attack_root + "/" + attack + "/LInf{}/A{}.pt".format(epsilon,i))

            

            # if epsilon > 5:
            #     print("Eps:", epsilon.item())
            #     print("Max Pert", torch.max(scaled_pert).item())
            #     print("Max Adv:", torch.max(adv_image).item())
            #     print("Min Adv:", torch.min(adv_image).item())
            #     print("Adv Mean:", torch.mean(adv_image.view(-1)).item())
            #     print("Img Mean:", torch.mean(image.view(-1)).item())
            #     from torchvision.utils import save_image
            #     save_image(adv_image.view(1,28,28)/255, "TEMP_adv.png")
            #     save_image(image.view(1,28,28)/255, "TEMP_img.png")
            #     save_image(scaled_pert.view(1,28,28)/255, "TEMP_pert.png")
            #     save_image(torch.clamp(image * 0, 0, 255).view(1,28,28)/255, "TEMP_black.png")
            #     save_image(torch.clamp(image + 500, 0, 255).view(1,28,28)/255, "TEMP_white.png")
                
            #     exit()
        
    
    # Join all images into one dataset
    for epsilon in epsilons:
        adv_images = torch.empty((  int(1e4),
                                        1,
                                        28, 
                                        28))

        for i in range(int(1e4)):
            adv_image = torch.load(attack_root + "/" + attack + "/LInf{}/A{}.pt".format(epsilon,i))
            if i ==0:
                print(adv_image.size())
            adv_images[i,:,:,:] = adv_image

        torch.save((adv_images, labels), attack_root + "/" + attack + "/LInf{}/adv_attacks.pt".format(epsilon))