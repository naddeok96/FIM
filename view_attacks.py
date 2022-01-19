from models.classes.first_layer_unitary_net  import FstLayUniNet
from adversarial_attacks import Attacker
from data_setup import Data
import numpy as np
import torch


set_name    = "MNIST"
from_ddp    = True
attack_type = "FGSM" 
gpu         = False
epsilons    = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0] # [round(x, 2) for x in np.linspace(0, 1, 7)]
model_name  = "cifar10_mobilenetv2_x1_0" # "lenet" # "cifar10_mobilenetv2_x1_0"
filename    = "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" # 
from_ddp    = True
attack_type = "FGSM" # "Gaussian_Noise" # "PGD"
gpu         = True
epsilons    = [round(x, 2) for x in np.linspace(0, 0.2, 5)]
print("Epsilons: ", epsilons)

if set_name == "MNIST":
    model_name  = "lenet" # "cifar10_mobilenetv2_x1_0"
    filename    = "models/pretrained/MNIST/lenet_w_acc_98.pt" # "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt"
else:
    model_name  = "cifar10_mobilenetv2_x1_0"
    filename    = "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt"



# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# Initialize data
data = Data(set_name = set_name, gpu = gpu, maxmin = True)

# # Load Attacker Net
attacker_net = FstLayUniNet(set_name, gpu = gpu,
                            U_filename = None,
                            model_name = model_name)
state_dict = torch.load(filename, map_location=torch.device('cpu'))

if from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")


# # torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.") # Remove prefixes if from DDP
attacker_net.load_state_dict(state_dict)
attacker_net.eval()

# Create an attacker
attacker = Attacker(attacker_net, data, gpu = gpu)

# Display Results
attacker.check_attack_perception(attack   = attack_type,
                                epsilons  = epsilons,
                                save_only = True)