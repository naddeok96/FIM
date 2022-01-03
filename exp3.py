# Imports
import os
import torch
import pickle5 as pickle
import matplotlib.pyplot as plt
import numpy as np
from data_setup import Data
from prettytable import PrettyTable
from adversarial_attacks import Attacker
from models.classes.first_layer_unitary_net  import FstLayUniNet
from tqdm import tqdm

print("Experimenet 3 -- Comparing Manifolds")

# Hyperparameters
gpu           = True
gpu_number    = "2"
set_name      = "MNIST" 
batch_size    = 5
prog_bar      = True

# Models
if set_name == "MNIST":
    reg1_model_name   = "lenet" # "cifar10_mobilenetv2_x1_0"
    reg1_net_filename = "models/pretrained/MNIST/lenet_w_acc_97.pt" # "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt" # "models/pretrained/MNIST/lenet_w_acc_97.pt"
    reg1_net_from_ddp = True
    reg1_net_acc    = 0.97

    reg2_model_name      = "lenet" # "cifar10_mobilenetv2_x1_0"
    reg2_net_filename    = "models/pretrained/MNIST/lenet_w_acc_98.pt"  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
    reg2_net_from_ddp    = True
    reg2_net_acc         = 0.93

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

else:
    reg1_model_name   = "cifar10_mobilenetv2_x1_0"
    reg1_net_filename = "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt" 
    reg1_net_from_ddp = True

    reg2_model_name      = "cifar10_mobilenetv2_x1_0"
    reg2_net_filename    = "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" 
    reg2_net_from_ddp    = True

    U_model_name        = "cifar10_mobilenetv2_x1_4"
    U_net_filename      = "models/pretrained/CIFAR10/U_cifar10_mobilenetv2_x1_4_w_acc_76.pt" 
    U_filename          = "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt" # 
    U_net_from_ddp      = True

# Declare which GPU PCI number to use
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


# Initialize table
table = PrettyTable()
# results = [epsilons]
# names   = ["Epsilons"]

# Initialize data
data = Data(gpu = gpu, set_name = set_name, maxmin = True, test_batch_size = batch_size)

# Load Networks
#-------------------------------------------------------------------------------------------------------------------------------------___#
# Load Reg1 Net
reg1_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = reg1_model_name)
reg1_net_state_dict = torch.load(reg1_net_filename, map_location=torch.device('cpu'))
if reg1_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(reg1_net_state_dict, "module.")
reg1_net.load_state_dict(reg1_net_state_dict)

reg1_net.eval()

# Load Reg2 Net
reg2_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = None,
                       model_name = reg2_model_name)
reg2_net_state_dict = torch.load(reg2_net_filename, map_location=torch.device('cpu'))
if reg2_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(reg2_net_state_dict, "module.")
reg2_net.load_state_dict(reg2_net_state_dict)
reg2_net.eval()

# Load UNet
U_net = FstLayUniNet(set_name, gpu =gpu,
                       U_filename = U_filename,
                       model_name = U_model_name)
U_net_state_dict = torch.load(U_net_filename, map_location=torch.device('cpu'))
if U_net_from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(U_net_state_dict, "module.")
U_net.load_state_dict(U_net_state_dict)
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

# Load AT-PGD Net
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
# Attacker net 
reg1_net_attacker = Attacker(reg1_net, data, gpu)

# Reg net 
reg2_net_attacker = Attacker(reg2_net, data, gpu)

# U net
U_net_attacker = Attacker(U_net, data, gpu)

# Distill
distill_attacker = Attacker(distill_net, data, gpu)

# Distill
adv_pgd_attacker = Attacker(adv_pgd_net, data, gpu)

# Declare Similarity Metric
cos_sim = torch.nn.CosineSimilarity()

# Cycle through data 
reg1_vs_reg2  = []
reg1_vs_U   = []
reg1_vs_distill = []
reg1_vs_adv_pgd = []
# reg2_vs_U   = []
for inputs, labels in tqdm (data.test_loader, desc="Batches Done...", disable=not prog_bar):
    if gpu:
        inputs, labels = inputs.cuda(), labels.cuda()

    # Generate max eigenvectors
    reg1_max_eigenvector, _ = reg1_net_attacker.get_max_eigenpair(inputs, labels)
    reg2_max_eigenvector, _ = reg2_net_attacker.get_max_eigenpair(inputs, labels)
    U_max_eigenvector  ,  _ =    U_net_attacker.get_max_eigenpair(inputs, labels)
    distill_max_eigenvector  ,  _ =    distill_attacker.get_max_eigenpair(inputs, labels)
    adv_pgd_max_eigenvector  ,  _ =    adv_pgd_attacker.get_max_eigenpair(inputs, labels)

    for i in range(inputs.size(0)):
        reg1_vs_reg2.append(np.abs(cos_sim(reg1_max_eigenvector[i].view(1, -1), reg2_max_eigenvector[i].view(1, -1)).item()))
        reg1_vs_U.append(np.abs(cos_sim(reg1_max_eigenvector[i].view(1, -1), U_max_eigenvector[i].view(1, -1)).item()))
        reg1_vs_distill.append(np.abs(cos_sim(reg1_max_eigenvector[i].view(1, -1), distill_max_eigenvector[i].view(1, -1)).item()))
        reg1_vs_adv_pgd.append(np.abs(cos_sim(reg1_max_eigenvector[i].view(1, -1), adv_pgd_max_eigenvector[i].view(1, -1)).item()))
        
        # reg2_vs_U.append(np.abs(cos_sim(reg2_max_eigenvector[i].view(1, -1), U_max_eigenvector[i].view(1, -1)).item()))
        # print("Reg1 vs Reg2 \t", cos_sim(reg1_max_eigenvector[i].view(1, -1), reg2_max_eigenvector[i].view(1, -1)).item()**2)
        # print("Reg1 vs U \t", cos_sim(reg1_max_eigenvector[i].view(1, -1), U_max_eigenvector[i].view(1, -1)).item()**2)
        # print("Reg2 vs U \t", cos_sim(reg2_max_eigenvector[i].view(1, -1), U_max_eigenvector[i].view(1, -1)).item()**2)

    # print("Breaking")
    # break

print("Black Box\t\t\t", np.mean(reg1_vs_reg2), np.std(reg1_vs_reg2))
print("U\t\t\t",         np.mean(reg1_vs_U), np.std(reg1_vs_U))
print("Distill\t\t\t",   np.mean(reg1_vs_distill), np.std(reg1_vs_distill))
print("AT-PGD\t\t\t",    np.mean(reg1_vs_adv_pgd), np.std(reg1_vs_adv_pgd))


results = [reg1_vs_reg2, reg1_vs_U, reg1_vs_distill, reg1_vs_adv_pgd]
 
fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(results, labels = ["No Defense", "OPP", "Distill", "AT-PGD"])
 
# show plot
plt.savefig("results/" + set_name + "/OSSA/manifold_box_plot.png")