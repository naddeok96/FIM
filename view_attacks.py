from models.classes.adjustable_lenet import AdjLeNet
from adversarial_attacks import Attacker
from data_setup import Data
import numpy as np
import torch


set_name = "MNIST"
epsilons = [round(x, 2) for x in np.linspace(0, 1, 5)]
print(epsilons)

# Initialize data
data = Data(set_name = set_name)

# Load Attacker LeNet
attacker_lenet = AdjLeNet(set_name = set_name)
attacker_lenet.load_state_dict(torch.load('models/pretrained/seed100_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
attacker_lenet.eval()

# Create an attacker
attacker = Attacker(attacker_lenet, 
                    data)


attacker.check_attack_perception(epsilons = epsilons)