from models.classes.first_layer_unitary_net  import FstLayUniNet
from adversarial_attacks import Attacker
from data_setup import Data
import numpy as np
import torch


set_name = "MNIST"
attack_type = "PGD"
gpu = True
epsilons = [round(x, 2) for x in np.linspace(0, 0.15, 5)]
print("Epsilons: ", epsilons)

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
                       model_name = "lenet")
state_dict = torch.load('models/pretrained/MNIST/lenet_w_acc_98.pt', map_location=torch.device('cpu'))

torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.") # Remove prefixes if from DDP
attacker_net.load_state_dict(state_dict)
attacker_net.eval()

# Create an attacker
attacker = Attacker(attacker_net, data, gpu = gpu)

# Display Results
attacker.check_attack_perception(attack   = attack_type,
                                epsilons  = epsilons,
                                save_only = True)