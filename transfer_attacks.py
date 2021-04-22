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
epsilons = [x/5 for x in range(31)]

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize table
table = PrettyTable()
table.add_column("Epsilons", epsilons)
results = [epsilons]
names   = ["Epsilons"]

# Initialize data
data = Data(gpu = gpu, set_name = set_name)

# Load Networks
#-------------------------------------------------------------------------------------------------------------------------------------___#
# # Load Attacker LeNet
attacker_lenet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
attacker_lenet.load_state_dict(torch.load('models/pretrained/MNIST/LeNet_Attacker_w_acc_98.pt', map_location=torch.device('cpu')))
attacker_lenet.eval()
attacker_lenet_acc = 0.98

# # Load LeNet
lenet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
lenet.load_state_dict(torch.load('models/pretrained/MNIST/LeNet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()
lenet_acc = 0.98

# Load WeakUnet
weakUnet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
weakUnet.load_state_dict(torch.load('models/pretrained/MNIST/WeakNet_w_acc_98.pt', map_location=torch.device('cpu')))
weakUnet.U = torch.load("models/pretrained/MNIST/U_WeakNet_w_acc_98.pt", map_location=torch.device('cpu'))
weakUnet.eval()
weakUnet_acc = 0.98

# Load Unet
Unet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet.load_state_dict(torch.load('models/pretrained/MNIST/UNet_w_acc_97.pt', map_location=torch.device('cpu')))
Unet.U = torch.load("models/pretrained/MNIST/U_UNet_w_acc_97.pt", map_location=torch.device('cpu'))
Unet.eval()
Unet_acc = 0.97

# Load Rnet
Rnet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Rnet.load_state_dict(torch.load('models/pretrained/MNIST/RNet_w_acc_96.pt', map_location=torch.device('cpu')))
Rnet.U = torch.load("models/pretrained/MNIST/U_RNet_w_acc_96.pt", map_location=torch.device('cpu'))
Rnet.eval()
Rnet_acc = 0.96

# Attack Networks
#-----------------------------------------------------------------------------------------------------------------------#
# Create an attacker
attacker = Attacker(attacker_lenet, 
                    data, 
                    gpu)

# # Get regular attack accuracies on attacker network
# print("Working on OSSA Attacks...")
# ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)                                          
# ossa_fool_ratio = attacker.get_fool_ratio(attacker_lenet_acc, ossa_accs)
# table.add_column("OSSA Fool Ratio", ossa_fool_ratio)
# results.append(ossa_fool_ratio)
# names.append("White Box Attack")


# # Get transfer attack accuracies for Lenet
# print("Working on LeNet OSSA Attacks...")
# lenet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
#                                                      transfer_network = lenet)                                          
# lenet_ossa_fool_ratio = attacker.get_fool_ratio(lenet_acc, lenet_ossa_accs)
# table.add_column("LeNet OSSA Fool Ratio", lenet_ossa_fool_ratio)
# results.append(lenet_ossa_fool_ratio)
# names.append("LeNet")

# Weak Unet 
print("Working on Weak UNet OSSA Attacks...")
weakUnet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = weakUnet)                                              
weakUnet_ossa_fool_ratio = attacker.get_fool_ratio(weakUnet_acc, weakUnet_ossa_accs)
table.add_column("Weak UNet OSSA Fool Ratio", weakUnet_ossa_fool_ratio)
results.append(weakUnet_ossa_fool_ratio)
names.append("weakUnet")

# Unet 
print("Working on UNet OSSA Attacks...")
Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet)                                              
Unet_ossa_fool_ratio = attacker.get_fool_ratio(Unet_acc, Unet_ossa_accs)
table.add_column("UNet OSSA Fool Ratio", Unet_ossa_fool_ratio)
results.append(Unet_ossa_fool_ratio)
names.append("Unet")
del Unet

# Rnet 
print("Working on RNet OSSA Attacks...")
Rnet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Rnet)                                              
Rnet_ossa_fool_ratio = attacker.get_fool_ratio(Rnet_acc, Rnet_ossa_accs)
table.add_column("RNet OSSA Fool Ratio", Rnet_ossa_fool_ratio)
results.append(Rnet_ossa_fool_ratio)
names.append("Rnet")
del Rnet

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

    for i, result in enumerate(results):
        sheet.write(0, i, names[i])

        for j, value in enumerate(result):
            sheet.write(j + 1, i, value)

    wb.save('results/MNIST/transfer_attack_results.xls') 
