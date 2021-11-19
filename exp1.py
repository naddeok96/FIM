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
save_to_excel = True
gpu           = True
gpu_number    = "2"
set_name      = "MNIST"
model_name    = "lenet" # "cifar10_mobilenetv2_x1_0" #
attack_type   = "CW2"
batch_size    = 512
# epsilons      = [0.0, 0.3] 
epsilons      = np.linspace(0, 1.0, num=61)

attack_net_filename = "models/pretrained/MNIST/lenet_w_acc_97.pt"
attack_net_from_ddp = True
attacker_net_acc    = 0.98

reg_net_filename    = "models/pretrained/MNIST/lenet_w_acc_98.pt" 
reg_net_from_ddp    = True
reg_net_acc         = 0.94

U_net_filename      = "models/pretrained/MNIST/Control_lenet_w_acc_97.pt" 
U_filename          = "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt"
U_net_from_ddp      = False
U_net_acc           = 0.97

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

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
                       model_name = model_name)
attack_net_state_dict = torch.load(attack_net_filename, map_location=torch.device('cpu'))
if attack_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(attack_net_state_dict, "module.")
attacker_net.load_state_dict(attack_net_state_dict)

attacker_net.eval()

# Load Regular Net
reg_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = model_name)
reg_net_state_dict = torch.load(reg_net_filename, map_location=torch.device('cpu'))
if reg_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(reg_net_state_dict, "module.")
reg_net.load_state_dict(reg_net_state_dict)
reg_net.eval()


# Load UNet
U_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = U_filename,
                       model_name = model_name)
U_net_state_dict = torch.load(U_net_filename, map_location=torch.device('cpu'))
if U_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(U_net_state_dict, "module.")
U_net.load_state_dict(U_net_state_dict)
U_net.eval()
U_net.eval()


# Attack Networks
#-----------------------------------------------------------------------------------------------------------------------#
# Create an attacker
attacker = Attacker(attacker_net, data, gpu)

print(attack_type + " attacks being performed...")

# Get regular attack accuracies on attacker network
# print("Working on White Box Attacks...")
# white_box_accs  = attacker.get_attack_accuracy(attack = attack_type, epsilons = epsilons)                                          
# ossa_fool_ratio = attacker.get_fool_ratio(attacker_net_acc, ossa_accs)
# table.add_column("OSSA Fool Ratio", ossa_fool_ratio)
# results.append(ossa_fool_ratio)
# names.append("White Box Attack")

# Reg net 
print("Working on Black Box Attacks...")
reg_net_attack_accs  = attacker.get_attack_accuracy(attack = attack_type,
                                            epsilons = epsilons,
                                            transfer_network = reg_net)                                              
reg_net_fool_ratio = attacker.get_fool_ratio(reg_net_acc, reg_net_attack_accs)
table.add_column("RegNet Fool Ratio", reg_net_fool_ratio)
results.append(reg_net_fool_ratio)
names.append("RegNet")

# Unet 
print("Working on Unet Attacks...")
U_net_attack_accs  = attacker.get_attack_accuracy(attack = attack_type,
                                           epsilons = epsilons,
                                           transfer_network = U_net)                                              
U_net_fool_ratio = attacker.get_fool_ratio(U_net_acc, U_net_attack_accs)
table.add_column("Unet Fool Ratio", U_net_fool_ratio)
results.append(U_net_fool_ratio)
names.append("Unet")

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

    wb.save('results/' + set_name + '/' + attack_type + '/' + 'UvsNoU_attack_results.xls') 
