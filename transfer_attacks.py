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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Initialize table
table = PrettyTable()
table.add_column("Epsilons", epsilons)

# Initialize data
data = Data(gpu = gpu, set_name = set_name)

# Load LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# UniLeNets

# Initialize
Unet1  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet2  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet3  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet4  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet5  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet6  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet7  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet8  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet9  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet10 = FstLayUniLeNet(set_name = set_name, gpu = gpu)

# Load Params
Unet1.load_state_dict(torch.load('models/pretrained/U1_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet2.load_state_dict(torch.load('models/pretrained/U2_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet3.load_state_dict(torch.load('models/pretrained/U3_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet4.load_state_dict(torch.load('models/pretrained/U4_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet5.load_state_dict(torch.load('models/pretrained/U5_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet6.load_state_dict(torch.load('models/pretrained/U6_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet7.load_state_dict(torch.load('models/pretrained/U7_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet8.load_state_dict(torch.load('models/pretrained/U8_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet9.load_state_dict(torch.load('models/pretrained/U9_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
Unet10.load_state_dict(torch.load('models/pretrained/U10_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))

# Load U's
with open("models/pretrained/" + str(0)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet1.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(1)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet2.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(2)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet3.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(3)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet4.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(4)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet5.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(5)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet6.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(6)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet7.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(7)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet8.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(8)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet9.U = pickle.load(input).type(torch.FloatTensor)
with open("models/pretrained/" + str(9)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet10.U = pickle.load(input).type(torch.FloatTensor)

# Evaluate
Unet1.eval()
Unet2.eval()
Unet3.eval()
Unet4.eval()
Unet5.eval()
Unet6.eval()
Unet7.eval()
Unet8.eval()
Unet9.eval()
Unet10.eval()

# Load Attacker Net
attacker_lenet = AdjLeNet(set_name = set_name)
attacker_lenet.load_state_dict(torch.load('models/pretrained/seed100_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
attacker_lenet.eval()

# Create an attacker
attacker = Attacker(attacker_lenet, 
                    data, 
                    gpu)

# Get regular attack accuracies on attacker network
print("Working on OSSA Attacks...")
ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)                                          
ossa_fool_ratio = attacker.get_fool_ratio(0.98, ossa_accs)
table.add_column("OSSA Fool Ratio", ossa_fool_ratio)

print("Working on FGSM Attacks...")
fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons)
fgsm_fool_ratio = attacker.get_fool_ratio(0.98, fgsm_accs)
table.add_column("FGSM Attack Accuracy", fgsm_fool_ratio)

# Get transfer attack accuracies for Lenet
print("Working on LeNet OSSA Attacks...")
lenet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = lenet)                                          
lenet_ossa_fool_ratio = attacker.get_fool_ratio(0.98, lenet_ossa_accs)
table.add_column("LeNet OSSA Fool Ratio", lenet_ossa_fool_ratio)

print("Working on LeNet FGSM Attacks...")
lenet_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = lenet)
lenet_fgsm_fool_ratio = attacker.get_fool_ratio(0.98, lenet_fgsm_accs)
table.add_column("LeNet FGSM Attack Accuracy", lenet_fgsm_fool_ratio)

# Get transfer attack accuracies for Unets

# Unet 1
print("Working on UNet1 OSSA Attacks...")
Unet1_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet1)                                              
Unet1_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet1_ossa_accs)
table.add_column("UNet1 OSSA Fool Ratio", Unet1_ossa_fool_ratio)

print("Working on UNet1 FGSM Attacks...")
Unet1_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet1)
Unet1_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet1_fgsm_accs)
table.add_column("UNet1 FGSM Attack Accuracy", Unet1_fgsm_fool_ratio)

# Unet 2
print("Working on UNet2 OSSA Attacks...")
Unet2_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet2)                                              
Unet2_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet2_ossa_accs)
table.add_column("UNet2 OSSA Fool Ratio", Unet2_ossa_fool_ratio)

print("Working on UNet2 FGSM Attacks...")
Unet2_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet2)
Unet2_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet2_fgsm_accs)
table.add_column("UNet2 FGSM Attack Accuracy", Unet2_fgsm_fool_ratio)

# Unet 3
print("Working on UNet3 OSSA Attacks...")
Unet3_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet3)                                              
Unet3_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet3_ossa_accs)
table.add_column("UNet3 OSSA Fool Ratio", Unet3_ossa_fool_ratio)

print("Working on UNet3 FGSM Attacks...")
Unet3_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet3)
Unet3_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet3_fgsm_accs)
table.add_column("UNet3 FGSM Attack Accuracy", Unet3_fgsm_fool_ratio)

# Unet 4
print("Working on UNet4 OSSA Attacks...")
Unet4_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet4)                                              
Unet4_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet4_ossa_accs)
table.add_column("UNet4 OSSA Fool Ratio", Unet4_ossa_fool_ratio)

print("Working on UNet4 FGSM Attacks...")
Unet4_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet4)
Unet4_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet4_fgsm_accs)
table.add_column("UNet4 FGSM Attack Accuracy", Unet4_fgsm_fool_ratio)

# Unet 5
print("Working on UNet5 OSSA Attacks...")
Unet5_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet5)                                              
Unet5_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet5_ossa_accs)
table.add_column("UNet5 OSSA Fool Ratio", Unet5_ossa_fool_ratio)

print("Working on UNet5 FGSM Attacks...")
Unet5_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet5)
Unet5_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet5_fgsm_accs)
table.add_column("UNet5 FGSM Attack Accuracy", Unet5_fgsm_fool_ratio)

# Unet 6
print("Working on UNet6 OSSA Attacks...")
Unet6_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet6)                                              
Unet6_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet6_ossa_accs)
table.add_column("UNet6 OSSA Fool Ratio", Unet6_ossa_fool_ratio)

print("Working on UNet6 FGSM Attacks...")
Unet6_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet6)
Unet6_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet6_fgsm_accs)
table.add_column("UNet6 FGSM Attack Accuracy", Unet6_fgsm_fool_ratio)

# Unet 7
print("Working on UNet7 OSSA Attacks...")
Unet7_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet7)                                              
Unet7_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet7_ossa_accs)
table.add_column("UNet7 OSSA Fool Ratio", Unet7_ossa_fool_ratio)

print("Working on UNet7 FGSM Attacks...")
Unet7_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet7)
Unet7_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet7_fgsm_accs)
table.add_column("UNet7 FGSM Attack Accuracy", Unet7_fgsm_fool_ratio)

# Unet 8
print("Working on UNet8 OSSA Attacks...")
Unet8_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet8)                                              
Unet8_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet8_ossa_accs)
table.add_column("UNet8 OSSA Fool Ratio", Unet8_ossa_fool_ratio)

print("Working on UNet8 FGSM Attacks...")
Unet8_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet8)
Unet8_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet8_fgsm_accs)
table.add_column("UNet8 FGSM Attack Accuracy", Unet8_fgsm_fool_ratio)

# Unet 9
print("Working on UNet9 OSSA Attacks...")
Unet9_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet9)                                              
Unet9_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet9_ossa_accs)
table.add_column("UNet9 OSSA Fool Ratio", Unet9_ossa_fool_ratio)

print("Working on UNet9 FGSM Attacks...")
Unet9_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet9)
Unet9_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet9_fgsm_accs)
table.add_column("UNet9 FGSM Attack Accuracy", Unet9_fgsm_fool_ratio)

# Unet 10
print("Working on UNet10 OSSA Attacks...")
Unet10_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet10)                                              
Unet10_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet10_ossa_accs)
table.add_column("UNet10 OSSA Fool Ratio", Unet10_ossa_fool_ratio)

print("Working on UNet10 FGSM Attacks...")
Unet10_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet10)
Unet10_fgsm_fool_ratio = attacker.get_fool_ratio(0.95, Unet10_fgsm_accs)
table.add_column("UNet10 FGSM Attack Accuracy", Unet10_fgsm_fool_ratio)

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
               ossa_fool_ratio, fgsm_fool_ratio,
               lenet_ossa_fool_ratio, lenet_fgsm_fool_ratio,
               Unet1_ossa_fool_ratio, Unet1_fgsm_fool_ratio,
               Unet2_ossa_fool_ratio, Unet2_fgsm_fool_ratio,
               Unet3_ossa_fool_ratio, Unet3_fgsm_fool_ratio,
               Unet4_ossa_fool_ratio, Unet4_fgsm_fool_ratio,
               Unet5_ossa_fool_ratio, Unet5_fgsm_fool_ratio,
               Unet6_ossa_fool_ratio, Unet6_fgsm_fool_ratio,
               Unet7_ossa_fool_ratio, Unet7_fgsm_fool_ratio,
               Unet8_ossa_fool_ratio, Unet8_fgsm_fool_ratio,
               Unet9_ossa_fool_ratio, Unet9_fgsm_fool_ratio,
               Unet10_ossa_fool_ratio, Unet10_fgsm_fool_ratio]
    names = ["epsilons", 
             "ossa_fool_ratio", "fgsm_fool_ratio",
             "lenet_ossa_fool_ratio", "lenet_fgsm_fool_ratio", 
             "Unet1_ossa_fool_ratio", "Unet1_fgsm_fool_ratio",
             "Unet2_ossa_fool_ratio", "Unet2_fgsm_fool_ratio",
             "Unet3_ossa_fool_ratio", "Unet3_fgsm_fool_ratio",
             "Unet4_ossa_fool_ratio", "Unet4_fgsm_fool_ratio",
             "Unet5_ossa_fool_ratio", "Unet5_fgsm_fool_ratio",
             "Unet6_ossa_fool_ratio", "Unet6_fgsm_fool_ratio",
             "Unet7_ossa_fool_ratio", "Unet7_fgsm_fool_ratio",
             "Unet8_ossa_fool_ratio", "Unet8_fgsm_fool_ratio",
             "Unet9_ossa_fool_ratio", "Unet9_fgsm_fool_ratio",
             "Unet10_ossa_fool_ratio", "Unet10_fgsm_fool_ratio"]

    for i, result in enumerate(results):
        sheet.write(0, i, names[i])

        for j, value in enumerate(result):
            sheet.write(j + 1, i, value)

    wb.save('10_Unet_transfer_attack_results.xls') 
