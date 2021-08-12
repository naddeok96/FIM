# Imports
import os
import torch
import pickle5 as pickle
import numpy as np
from data_setup import Data
from prettytable import PrettyTable
from adversarial_attacks import Attacker
from models.classes.first_layer_unitary_net  import FstLayUniNet


# Hyperparameters
save_to_excel = False
gpu = False
set_name = "MNIST"
attack_type = "EOT"
batch_size = 10
epsilons = np.linspace(0, 0.15, num=61)

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Initialize table
table = PrettyTable()
table.add_column("Epsilons", epsilons)
results = [epsilons]
names   = ["Epsilons"]

# Initialize data
data = Data(gpu = gpu, set_name = set_name, maxmin = True, test_batch_size = batch_size)

# Load Networks
#-------------------------------------------------------------------------------------------------------------------------------------___#
# # Load Attacker Net
attacker_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = "cifar10_mobilenetv2_x1_0",
                       pretrained = False)
attacker_net.load_state_dict(torch.load('models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt', map_location=torch.device('cpu')))
attacker_net.eval()
attacker_net_acc = 0.91

# # Load Regular Net
reg_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = "cifar10_mobilenetv2_x1_4",
                       pretrained = True)
reg_net.eval()
reg_net_acc = 0.94

# # Load UNet
Unet = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = "models/pretrained/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt",
                       model_name = "cifar10_mobilenetv2_x1_4",
                       pretrained = False)
Unet.load_state_dict(torch.load('models/pretrained/CIFAR10/Ucifar10_mobilenetv2_x1_4_w_acc_78.pt', map_location=torch.device('cpu')))
Unet.eval()
Unet_acc = 0.78


# Attack Networks
#-----------------------------------------------------------------------------------------------------------------------#
# Create an attacker
attacker = Attacker(attacker_net, data, gpu)

print(attack_type + " attacks being performed...")

# Get regular attack accuracies on attacker network
print("Working on White Box Attacks...")
ossa_accs  = attacker.get_attack_accuracy(attack = attack_type, epsilons = epsilons)                                          
ossa_fool_ratio = attacker.get_fool_ratio(attacker_net_acc, ossa_accs)
table.add_column("OSSA Fool Ratio", ossa_fool_ratio)
results.append(ossa_fool_ratio)
names.append("White Box Attack")

# Reg net 
# print("Working on Black Box Attacks...")
# reg_net_ossa_accs  = attacker.get_attack_accuracy(attack = attack_type,
#                                                     epsilons = epsilons,
#                                                     transfer_network = reg_net)                                              
# reg_net_ossa_fool_ratio = attacker.get_fool_ratio(reg_net_acc, reg_net_ossa_accs)
# table.add_column("RegNet OSSA Fool Ratio", reg_net_ossa_fool_ratio)
# results.append(reg_net_ossa_fool_ratio)
# names.append("RegNet")

# # # Unet 
# print("Working on Unet Attacks...")
# Unet_ossa_accs  = attacker.get_attack_accuracy(attack = attack_type,
#                                                 epsilons = epsilons,
#                                                 transfer_network = Unet)                                              
# Unet_ossa_fool_ratio = attacker.get_fool_ratio(Unet_acc, Unet_ossa_accs)
# table.add_column("Unet OSSA Fool Ratio", Unet_ossa_fool_ratio)
# results.append(Unet_ossa_fool_ratio)
# names.append("Unet")

# Display
print(table)

# Excel Workbook Object is created 
if save_to_excel:
    from xlwt import Workbook  

    # Open Workbook
    wb = Workbook() 
    
    # Create sheet
    sheet = wb.add_sheet('Results') 

    for i, result in enumerate(results):
        sheet.write(0, i, names[i])

        for j, value in enumerate(result):
            sheet.write(j + 1, i, value)

    wb.save('results/CIFAR10/W' + attack_type + '_attack_results.xls') 
