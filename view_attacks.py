from models.classes.first_layer_unitary_net  import FstLayUniNet
from adversarial_attacks import Attacker
from data_setup import Data
import numpy as np
import torch


set_name = "CIFAR10"
attack_type = "EOT"
gpu = True
epsilons = [round(x, 2) for x in np.linspace(0, 0.4, 7)]
print(epsilons)

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Initialize data
data = Data(set_name = set_name, gpu = gpu, maxmin = True)

# # Load Attacker Net
attacker_net = FstLayUniNet(set_name, gpu = gpu,
                       U_filename = None,
                       model_name = "cifar10_mobilenetv2_x1_0",
                       pretrained = False)
attacker_net.load_state_dict(torch.load('models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt', map_location=torch.device('cpu')))
attacker_net.eval()

# Create an attacker
attacker = Attacker(attacker_net, data, gpu = gpu)

# Display Results
attacker.check_attack_perception(attack = attack_type,
                                epsilons = epsilons,
                                save_only=True)