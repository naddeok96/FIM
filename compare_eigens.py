# Imports
from models.classes.first_layer_unitary_dense_net import FstLayUniDenseNet
from models.classes.adjustable_lenet import AdjLeNet
from adversarial_attacks import Attacker
from prettytable import PrettyTable, ALL
from data_setup import Data
from academy import Academy
import numpy as np

# Hyperparameters
set_name = "MNIST"
gpu = False
num_images = 5
eig_vec_head_size = 3

# Initialize data
data = Data(gpu, set_name)

# LeNet
regnet = FstLayUniDenseNet(set_name = set_name,
                 pretrained_weights_filename = 'models/pretrained/mixed_U_dense_net_w_acc_98.pt')

# UNet
Unet = FstLayUniDenseNet(set_name = set_name,
                      pretrained_weights_filename = 'models/pretrained/mixed_U_dense_net_w_acc_98.pt',
                      pretrained_unitary_matrix_filename = 'models/pretrained/U_for_mixed_U_dense_net_w_acc_98.pt')

# Initialize attackers
regnet_attacker = Attacker(regnet, data)
Unet_attacker = Attacker(Unet, data)

# Compare Eigensystems
table = PrettyTable(hrules = ALL)
table.field_names = ["Image Label", "Reg-Net Eigenvalue", "Reg-Net Eigenvector Head",
                                    "U-Net Eigenvalue", "U-Net Eigenvector Head"]

for image_index in range(num_images):
    regnet_eig_val_max, regnet_eig_vec_max = regnet_attacker.get_single_OSSA_attack(image_index = image_index,
                                                                                 eigens_only = True)

    Unet_eig_val_max, Unet_eig_vec_max = Unet_attacker.get_single_OSSA_attack(image_index = image_index,
                                                                              eigens_only = True)

    # Convert to Scientific Notation
    regnet_eig_val_max = "{:.2e}".format(regnet_eig_val_max.item())
    Unet_eig_val_max = "{:.2e}".format(Unet_eig_val_max.item())
    regnet_eig_vec_head = ["{:.2e}".format(x) for x in regnet_eig_vec_max.squeeze().squeeze().tolist()[0:eig_vec_head_size]]
    Unet_eig_vec_head = ["{:.2e}".format(x) for x in Unet_eig_vec_max.squeeze().squeeze().tolist()[0:eig_vec_head_size]]

    # Load Table
    for j in range(eig_vec_head_size  + 1):
        if j == 0:
            table.add_row([image_index, regnet_eig_val_max, regnet_eig_vec_head[j], 
                                        Unet_eig_val_max,  Unet_eig_vec_head[j]])
        elif j != eig_vec_head_size:
            table.add_row(["-", "-", regnet_eig_vec_head[j], "-", Unet_eig_vec_head[j]])
        else:
            table.add_row(["-", "-", ":" , "-", ":"])
            table.add_row([("/"*int(np.floor(len(table.field_names[i])/2))) + ("/"*int(np.floor(len(table.field_names[i])/2))) for i in range(len(table.field_names))])

print(table)
