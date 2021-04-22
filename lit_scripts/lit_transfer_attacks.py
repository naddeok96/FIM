# Imports
import os
import torch
import pickle5 as pickle
from data_setup import Data
from prettytable import PrettyTable
from adversarial_attacks import Attacker
from models.classes.adjustable_lenet import AdjLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet
from models.classes.lit_lenet import LitLeNet

# Hyperparameters
save_to_excel = True
gpu = True
set_name = "MNIST"
epsilons = [x/5 for x in range(31)]

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Initialize table
table = PrettyTable()
table.add_column("Epsilons", epsilons)

# Initialize data
data = Data(gpu = gpu, set_name = set_name)

# Initalize Models
attacker_net = LitLeNet.load_from_checkpoint(
    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_no_U_attacker-val_acc=0.99.ckpt',
    set_name=set_name,
    learning_rate = 0.01, 
    momentum = 0.9, 
    weight_decay = 0.001,
    batch_size=256)

lenet = LitLeNet.load_from_checkpoint(
    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_no_U-val_acc=0.98.ckpt',
    set_name=set_name,
    learning_rate = 0.01, 
    momentum = 0.9, 
    weight_decay = 0.001,
    batch_size=256)
 
weak_Unet = LitLeNet.load_from_checkpoint(
    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_weak_U-val_acc=0.98-v1.ckpt',
    set_name=set_name,
    learning_rate = 0.01, 
    momentum = 0.9, 
    weight_decay = 0.001,
    batch_size=256)   
weak_Unet.load_orthogonal_matrix("models/pretrained/high_R_U.pkl")

Unet = LitLeNet.load_from_checkpoint(
    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_mini_standard_U-val_acc=0.97.ckpt',
    set_name=set_name,
    learning_rate = 0.01, 
    momentum = 0.9, 
    weight_decay = 0.001,
    batch_size=256)   
Unet.load_orthogonal_matrix("models/pretrained/LeNet_MNIST_mini_standard_U.pkl")

Rnet = LitLeNet.load_from_checkpoint(
    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_random_U-val_acc=0.95.ckpt',
    set_name=set_name,
    learning_rate = 0.01, 
    momentum = 0.9, 
    weight_decay = 0.001,
    batch_size=256)   
Unet.load_orthogonal_matrix("models/pretrained/LeNet_MNIST_random_U.pkl")


# Create an attacker
attacker = Attacker(attacker_net, 
                    data, 
                    gpu)


# Regular Net 
print("Working on OSSA Attacks...")
ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)                                              
ossa_fool_ratio = attacker.get_fool_ratio(0.99, ossa_accs)
table.add_column("OSSA Fool Ratio", ossa_fool_ratio)

# Lenet 
print("Working on LeNet OSSA Attacks...")
lenet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = lenet)                                              
lenet_ossa_fool_ratio = attacker.get_fool_ratio(0.98, lenet_ossa_accs)
table.add_column("LeNet OSSA Fool Ratio", lenet_ossa_fool_ratio)

# Weak Lenet 
print("Working on Weak UNet OSSA Attacks...")
weak_Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = weak_Unet)                                              
weak_Unet_ossa_fool_ratio = attacker.get_fool_ratio(0.98, weak_Unet_ossa_accs)
table.add_column("Weak UNet OSSA Fool Ratio", weak_Unet_ossa_fool_ratio)

# Unet 
print("Working on UNet OSSA Attacks...")
Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet)                                              
Unet_ossa_fool_ratio = attacker.get_fool_ratio(0.97, Unet_ossa_accs)
table.add_column("UNet OSSA Fool Ratio", Unet_ossa_fool_ratio)

# Rnet 
print("Working on RNet OSSA Attacks...")
Rnet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Rnet)                                              
Rnet_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Rnet_ossa_accs)
table.add_column("RNet OSSA Fool Ratio", Rnet_ossa_fool_ratio)

# Display
print(table)

# Excel Workbook Object is created 
if save_to_excel:
    import xlwt 
    from xlwt import Workbook  

    # Open Workbook
    wb = Workbook() 
    
    # Create sheet
    sheet = wb.add_sheet('Results') 

    # Write out each peak and data
    results = [epsilons, 
               ossa_fool_ratio,
               lenet_ossa_fool_ratio,
               Rnet_ossa_fool_ratio,
               Unet_ossa_fool_ratio,
               weak_Unet_ossa_fool_ratio]
    names = ["epsilons", 
             "White-Box",
             "LeNet",
             "Random Unet", 
             "Unet",
             "Weak Unet"]

    for i, result in enumerate(results):
        sheet.write(0, i, names[i])

        for j, value in enumerate(result):
            sheet.write(j + 1, i, value)

    wb.save('mini_Unet_transfer_attack_results.xls') 
