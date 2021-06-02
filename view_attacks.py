from models.classes.first_layer_unitary_net  import FstLayUniNet
from adversarial_attacks import Attacker
from data_setup import Data
import numpy as np
import torch


set_name = "CIFAR10"
epsilons = [round(x, 2) for x in np.linspace(0, 0.2, 7)]
print(epsilons)

# Initialize data
data = Data(set_name = set_name)

# # Load Attacker Net
attacker_net = FstLayUniNet(set_name, gpu = False,
                       U_filename = None,
                       model_name = "cifar10_mobilenetv2_x1_0",
                       pretrained = False)
attacker_net.load_state_dict(torch.load('models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt', map_location=torch.device('cpu')))
attacker_net.eval()

# Create an attacker
attacker = Attacker(attacker_net, data)

# Display Results
attacker.check_attack_perception(epsilons = epsilons)