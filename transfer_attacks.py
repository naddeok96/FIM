# Imports
import os
import torch
import pickle5 as pickle
from data_setup import Data
from prettytable import PrettyTable
from adversarial_attacks import Attacker
from models.classes.first_layer_unitary_net  import FstLayUniNet

# Hyperparameters
save_to_excel = True
gpu = True
set_name = "CIFAR10"
epsilons = [x/5 for x in range(31)]

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Initialize table
table = PrettyTable()
table.add_column("Epsilons", epsilons)
results = [epsilons]
names   = ["Epsilons"]

# Initialize data
data = Data(gpu = gpu, set_name = set_name)

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

# Get regular attack accuracies on attacker network
print("Working on OSSA Attacks...")
ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)                                          
ossa_fool_ratio = attacker.get_fool_ratio(attacker_net_acc, ossa_accs)
table.add_column("OSSA Fool Ratio", ossa_fool_ratio)
results.append(ossa_fool_ratio)
names.append("White Box Attack")

# Unet 
# print("Working on Unet OSSA Attacks...")
# Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
#                                                      transfer_network = Unet)                                              
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

    wb.save('results/CIFAR10/transfer_attack_results.xls') 
