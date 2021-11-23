# Imports
import os
import torch
import pickle5 as pickle
import numpy as np
from data_setup import Data
from prettytable import PrettyTable
from adversarial_attacks import Attacker
from models.classes.first_layer_unitary_net  import FstLayUniNet

print("Experiment 2: Multiple Defenses")

# Hyperparameters
save_to_excel = True
gpu           = True
gpu_number    = "1"
set_name      = "CIFAR10" # "MNIST" # "CIFAR10" # 
attack_type   = "CW2"
batch_size    = 124
# epsilons      = [0.0, 0.3] 
epsilons      = np.linspace(0, 1.0, num=61)

if set_name == "MNIST":
    attack_model_name   = "lenet" # "cifar10_mobilenetv2_x1_0"
    attack_net_filename = "models/pretrained/MNIST/lenet_w_acc_97.pt" # "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt" # "models/pretrained/MNIST/lenet_w_acc_97.pt"
    attack_net_from_ddp = True
    attacker_net_acc    = 0.97

    reg_model_name      = "lenet" # "cifar10_mobilenetv2_x1_0"
    reg_net_filename    = "models/pretrained/MNIST/lenet_w_acc_98.pt"  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
    reg_net_from_ddp    = True
    reg_net_acc         = 0.98

    U_model_name        = "lenet" # "cifar10_mobilenetv2_x1_4"
    U_net_filename      =  "models/pretrained/MNIST/U_lenet_w_acc_94.pt"  # "models/pretrained/CIFAR10/U_cifar10_mobilenetv2_x1_4_w_acc_76.pt" # "models/pretrained/MNIST/Control_lenet_w_acc_97.pt" 
    U_filename          = "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt" # "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt" # 
    U_net_from_ddp      = True
    U_net_acc           = 0.94

    distill_model_name   = "lenet" # "cifar10_mobilenetv2_x1_0"
    distill_net_filename = "models/pretrained/MNIST/distilled_20_lenet_w_acc_94.pt"  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
    distill_net_from_ddp = True
    distill_net_acc      = 0.94

    adv_pgd_model_name   = "lenet" # "cifar10_mobilenetv2_x1_0"
    adv_pgd_net_filename = "models/pretrained/MNIST/PGD_15_lenet_w_acc_97.pt"  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
    adv_pgd_net_from_ddp = True
    adv_pgd_net_acc      = 0.97

elif set_name == "CIFAR10":
    attack_model_name   = "cifar10_mobilenetv2_x1_0"
    attack_net_filename = "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt" 
    attack_net_from_ddp = True
    attacker_net_acc    = 0.1

    reg_model_name      = "cifar10_mobilenetv2_x1_0"
    reg_net_filename    = "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" 
    reg_net_from_ddp    = True
    reg_net_acc         = 0.93

    U_model_name        = "cifar10_mobilenetv2_x1_4"
    U_net_filename      = "models/pretrained/CIFAR10/U_cifar10_mobilenetv2_x1_4_w_acc_76.pt" 
    U_filename          = "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt" # 
    U_net_from_ddp      = True
    U_net_acc           = 0.76

    distill_model_name   = "cifar10_mobilenetv2_x1_0"
    distill_net_filename = "models/pretrained/CIFAR10/distilled_20_cifar10_mobilenetv2_x1_0_w_acc_89.pt" 
    distill_net_from_ddp = True
    distill_net_acc      = 0.89

    adv_pgd_model_name   = "cifar10_mobilenetv2_x1_0"
    adv_pgd_net_filename = "models/pretrained/CIFAR10/TEMP_PGD_15_cifar10_mobilenetv2_x1_0_w_acc_72.pt"
    adv_pgd_net_from_ddp = True
    adv_pgd_net_acc      = 0.72

else:
    print("Invalid set name")
    exit()

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
# Load Attacker Net
attacker_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = attack_model_name)
attack_net_state_dict = torch.load(attack_net_filename, map_location=torch.device('cpu'))
if attack_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(attack_net_state_dict, "module.")
attacker_net.load_state_dict(attack_net_state_dict)

attacker_net.eval()

# Load Regular Net
reg_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = reg_model_name)
reg_net_state_dict = torch.load(reg_net_filename, map_location=torch.device('cpu'))
if reg_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(reg_net_state_dict, "module.")
reg_net.load_state_dict(reg_net_state_dict)
reg_net.eval()

# Load UNet
U_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = U_filename,
                       model_name = U_model_name)
U_net_state_dict = torch.load(U_net_filename, map_location=torch.device('cpu'))
if U_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(U_net_state_dict, "module.")
U_net.load_state_dict(U_net_state_dict)
U_net.eval()
U_net.eval()

# Load Distill Net
distill_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = distill_model_name)
distill_net_state_dict = torch.load(distill_net_filename, map_location=torch.device('cpu'))
if distill_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(distill_net_state_dict, "module.")
distill_net.load_state_dict(distill_net_state_dict)
distill_net.eval()

# Load Distill Net
adv_pgd_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = adv_pgd_model_name)
adv_pgd_net_state_dict = torch.load(adv_pgd_net_filename, map_location=torch.device('cpu'))
if adv_pgd_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(adv_pgd_net_state_dict, "module.")
adv_pgd_net.load_state_dict(adv_pgd_net_state_dict)
adv_pgd_net.eval()


# Attack Networks
#-----------------------------------------------------------------------------------------------------------------------#
# Create an attacker
attacker = Attacker(attacker_net, data, gpu)

print(attack_type + " attacks being performed...")

# Get regular attack accuracies on attacker network
print("Working on White Box Attacks...")
white_box_accs  = attacker.get_attack_accuracy(attack = attack_type, epsilons = epsilons)                                          
white_box_fool_ratio = attacker.get_fool_ratio(attacker_net_acc, white_box_accs)
table.add_column("White Box Fool Ratio", white_box_fool_ratio)
results.append(white_box_fool_ratio)
names.append("White Box Attack")

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

# Distill Net
print("Working on Distill Net Attacks...")
distill_net_attack_accs  = attacker.get_attack_accuracy(attack = attack_type,
                                           epsilons = epsilons,
                                           transfer_network = distill_net)                                              
distill_net_fool_ratio = attacker.get_fool_ratio(distill_net_acc, distill_net_attack_accs)
table.add_column("Distill Net Fool Ratio", distill_net_fool_ratio)
results.append(distill_net_fool_ratio)
names.append("Distill Net")


# AT-PGD Net
print("Working on AT-PGD Net Attacks...")
adv_pgd_net_attack_accs  = attacker.get_attack_accuracy(attack = attack_type,
                                           epsilons = epsilons,
                                           transfer_network = adv_pgd_net)                                              
adv_pgd_net_fool_ratio = attacker.get_fool_ratio(adv_pgd_net_acc, adv_pgd_net_attack_accs)
table.add_column("AT-PGD Net Fool Ratio", adv_pgd_net_fool_ratio)
results.append(adv_pgd_net_fool_ratio)
names.append("AT-PGD Net")

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

    wb.save('results/' + set_name + '/' + attack_type + '/' + 'defense_comparison.xls') 
