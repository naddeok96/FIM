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
attacker_lenet = FstLayUniLeNet(set_name = set_name, gpu = gpu,
                                num_kernels_layer1 = 12, 
                                num_kernels_layer2 = 32, 
                                num_kernels_layer3 = 240,
                                num_nodes_fc_layer = 168)
attacker_lenet.load_state_dict(torch.load('models/pretrained/MNIST/Large_LeNet_Attacker_w_acc_98.pt', map_location=torch.device('cpu')))
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

# Load Unet1
Unet1 = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet1.load_state_dict(torch.load('models/pretrained/MNIST2/Unet1_w_acc_95.pt', map_location=torch.device('cpu')))
Unet1.U = torch.load("models/pretrained/MNIST2/U_Unet1_w_acc_95.pt", map_location=torch.device('cpu'))
Unet1.eval()
Unet1_acc = 0.95

# Load Unet2
Unet2 = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet2.load_state_dict(torch.load('models/pretrained/MNIST2/Unet2_w_acc_95.pt', map_location=torch.device('cpu')))
Unet2.U = torch.load("models/pretrained/MNIST2/U_Unet2_w_acc_95.pt", map_location=torch.device('cpu'))
Unet2.eval()
Unet2_acc = 0.95

# Load Unet3
Unet3 = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet3.load_state_dict(torch.load('models/pretrained/MNIST2/Unet3_w_acc_95.pt', map_location=torch.device('cpu')))
Unet3.U = torch.load("models/pretrained/MNIST2/U_Unet3_w_acc_95.pt", map_location=torch.device('cpu'))
Unet3.eval()
Unet3_acc = 0.95

# Load Rnet1
Rnet1 = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Rnet1.load_state_dict(torch.load('models/pretrained/MNIST2/Rnet1_w_acc_94.pt', map_location=torch.device('cpu')))
Rnet1.U = torch.load("models/pretrained/MNIST2/U_Rnet1_w_acc_94.pt", map_location=torch.device('cpu'))
Rnet1.eval()
Rnet1_acc = 0.94

# Load Rnet2
Rnet2 = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Rnet2.load_state_dict(torch.load('models/pretrained/MNIST2/Rnet2_w_acc_94.pt', map_location=torch.device('cpu')))
Rnet2.U = torch.load("models/pretrained/MNIST2/U_Rnet2_w_acc_94.pt", map_location=torch.device('cpu'))
Rnet2.eval()
Rnet2_acc = 0.94

# Load Rnet3
Rnet3 = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Rnet3.load_state_dict(torch.load('models/pretrained/MNIST2/Rnet3_w_acc_94.pt', map_location=torch.device('cpu')))
Rnet3.U = torch.load("models/pretrained/MNIST2/U_Rnet3_w_acc_94.pt", map_location=torch.device('cpu'))
Rnet3.eval()
Rnet3_acc = 0.94

# Attack Networks
#-----------------------------------------------------------------------------------------------------------------------#
# Create an attacker
attacker = Attacker(attacker_lenet, 
                    data, 
                    gpu)

# Get regular attack accuracies on attacker network
print("Working on OSSA Attacks...")
ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)                                          
ossa_fool_ratio = attacker.get_fool_ratio(attacker_lenet_acc, ossa_accs)
table.add_column("OSSA Fool Ratio", ossa_fool_ratio)
results.append(ossa_fool_ratio)
names.append("White Box Attack")


# # Get transfer attack accuracies for Lenet
# print("Working on LeNet OSSA Attacks...")
# lenet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
#                                                      transfer_network = lenet)                                          
# lenet_ossa_fool_ratio = attacker.get_fool_ratio(lenet_acc, lenet_ossa_accs)
# table.add_column("LeNet OSSA Fool Ratio", lenet_ossa_fool_ratio)
# results.append(lenet_ossa_fool_ratio)
# names.append("LeNet")

# # Weak Unet 
# print("Working on Weak UNet OSSA Attacks...")
# weakUnet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
#                                                      transfer_network = weakUnet)                                              
# weakUnet_ossa_fool_ratio = attacker.get_fool_ratio(weakUnet_acc, weakUnet_ossa_accs)
# table.add_column("Weak UNet OSSA Fool Ratio", weakUnet_ossa_fool_ratio)
# results.append(weakUnet_ossa_fool_ratio)
# names.append("weakUnet")

# Unet 
print("Working on Unet1 OSSA Attacks...")
Unet1_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet1)                                              
Unet1_ossa_fool_ratio = attacker.get_fool_ratio(Unet1_acc, Unet1_ossa_accs)
table.add_column("Unet1 OSSA Fool Ratio", Unet1_ossa_fool_ratio)
results.append(Unet1_ossa_fool_ratio)
names.append("Unet1")

# Unet2 
print("Working on Unet2 OSSA Attacks...")
Unet2_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet2)                                              
Unet2_ossa_fool_ratio = attacker.get_fool_ratio(Unet2_acc, Unet2_ossa_accs)
table.add_column("Unet2 OSSA Fool Ratio", Unet2_ossa_fool_ratio)
results.append(Unet2_ossa_fool_ratio)
names.append("Unet2")

# Unet3 
print("Working on Unet3 OSSA Attacks...")
Unet3_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet3)                                              
Unet3_ossa_fool_ratio = attacker.get_fool_ratio(Unet3_acc, Unet3_ossa_accs)
table.add_column("Unet3 OSSA Fool Ratio", Unet3_ossa_fool_ratio)
results.append(Unet3_ossa_fool_ratio)
names.append("Unet3")

# Rnet1 
print("Working on Rnet1 OSSA Attacks...")
Rnet1_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Rnet1)                                              
Rnet1_ossa_fool_ratio = attacker.get_fool_ratio(Rnet1_acc, Rnet1_ossa_accs)
table.add_column("Rnet1 OSSA Fool Ratio", Rnet1_ossa_fool_ratio)
results.append(Rnet1_ossa_fool_ratio)
names.append("Rnet1")

# Rnet2 
print("Working on Rnet2 OSSA Attacks...")
Rnet2_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Rnet2)                                              
Rnet2_ossa_fool_ratio = attacker.get_fool_ratio(Rnet2_acc, Rnet2_ossa_accs)
table.add_column("Rnet2 OSSA Fool Ratio", Rnet2_ossa_fool_ratio)
results.append(Rnet2_ossa_fool_ratio)
names.append("Rnet2")

# Rnet3 
print("Working on Rnet3 OSSA Attacks...")
Rnet3_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Rnet3)                                              
Rnet3_ossa_fool_ratio = attacker.get_fool_ratio(Rnet3_acc, Rnet3_ossa_accs)
table.add_column("Rnet3 OSSA Fool Ratio", Rnet3_ossa_fool_ratio)
results.append(Rnet3_ossa_fool_ratio)
names.append("Rnet3")


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

    wb.save('results/MNIST/Large_LeNet_transfer_attack_results.xls') 
