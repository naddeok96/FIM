# Imports
import os
import torch
import pickle5 as pickle
from data_setup import Data
from prettytable import PrettyTable
from adversarial_attacks import Attacker
from models.classes.adjustable_lenet import AdjLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet

# Hyperparameters
save_to_excel = True
gpu = True
set_name = "MNIST"
epsilons = [x/5 for x in range(61)]

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Initialize table
table = PrettyTable()
table.add_column("Epsilons", epsilons)

# Initialize data
data = Data(gpu = gpu, set_name = set_name)

# # Load LeNet
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





# lenet = AdjLeNet(set_name = set_name)
# lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
# lenet.eval()

# # UniNet
# Unet  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
# Unet.load_state_dict(torch.load('models/pretrained/U1_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
# with open("models/pretrained/" + str(0)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
#     Unet.U = pickle.load(input).type(torch.FloatTensor)
# Unet.eval()

# # Weak Uninet
# weak_Unet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
# weak_Unet.load_state_dict(torch.load('models/pretrained/High_R_Unet_w_acc_98.pt', map_location=torch.device('cpu')))
# with open("models/pretrained/high_R_U.pkl", 'rb') as input:
#     weak_Unet.U = pickle.load(input).type(torch.FloatTensor)
# weak_Unet.eval()

# # Load MiniUnet
# miniUnet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
# miniUnet.load_state_dict(torch.load('models/pretrained/MNIST_miniU_lenet_w_acc_97.pt', map_location=torch.device('cpu')))
# miniUnet.U = torch.load("models/pretrained/U_MNIST_miniU_lenet_w_acc_97.pt", map_location=torch.device('cpu'))
# miniUnet.eval()

# # Load Attacker Net
# attacker_lenet = AdjLeNet(set_name = set_name)
# attacker_lenet.load_state_dict(torch.load('models/pretrained/seed100_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
# attacker_lenet.eval()

# Create an attacker
attacker = Attacker(attacker_lenet, 
                    data, 
                    gpu)

# # Get regular attack accuracies on attacker network
# print("Working on OSSA Attacks...")
# ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)                                          
# ossa_fool_ratio = attacker.get_fool_ratio(0.98, ossa_accs)
# table.add_column("OSSA Fool Ratio", ossa_fool_ratio)

# # Get transfer attack accuracies for Lenet
# print("Working on LeNet OSSA Attacks...")
# lenet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
#                                                      transfer_network = lenet)                                          
# lenet_ossa_fool_ratio = attacker.get_fool_ratio(0.98, lenet_ossa_accs)
# table.add_column("LeNet OSSA Fool Ratio", lenet_ossa_fool_ratio)

# # Unet 
# print("Working on UNet OSSA Attacks...")
# Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
#                                                      transfer_network = Unet)                                              
# Unet_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet_ossa_accs)
# table.add_column("UNet OSSA Fool Ratio", Unet_ossa_fool_ratio)

# # Weak Unet 
# print("Working on Weak UNet OSSA Attacks...")
# weak_Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
#                                                      transfer_network = weak_Unet)                                              
# weak_Unet_ossa_fool_ratio = attacker.get_fool_ratio(0.95, weak_Unet_ossa_accs)
# table.add_column("Weak UNet OSSA Fool Ratio", weak_Unet_ossa_fool_ratio)

# Mini Unet 
print("Working on Mini UNet OSSA Attacks...")
miniUnet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = miniUnet)                                              
miniUnet_ossa_fool_ratio = attacker.get_fool_ratio(0.97, miniUnet_ossa_accs)
table.add_column("Mini UNet OSSA Fool Ratio", miniUnet_ossa_fool_ratio)

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
            #    ossa_fool_ratio,
            #    lenet_ossa_fool_ratio,
            #    Unet_ossa_fool_ratio,
            #    weak_Unet_ossa_fool_ratio,
               miniUnet_ossa_fool_ratio]
    names = ["epsilons", 
            #  "ossa_fool_ratio",
            #  "lenet_ossa_fool_ratio",
            #  "Unet_ossa_fool_ratio", 
            #  "weak_Unet_ossa_fool_ratio",
             "mini_Unet_ossa_fool_ratio"]

    for i, result in enumerate(results):
        sheet.write(0, i, names[i])

        for j, value in enumerate(result):
            sheet.write(j + 1, i, value)

    wb.save('mini_Unet_transfer_attack_results.xls') 
