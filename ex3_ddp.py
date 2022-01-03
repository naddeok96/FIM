import os
import wandb
import torch
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import copy
import numpy as np
from prettytable import PrettyTable
import torch.distributed as dist
import torch.multiprocessing as mp
from adversarial_attacks import Attacker
from data_setup import Data
import torch.nn.functional as F
from models.classes.first_layer_unitary_net  import FstLayUniNet
from torch.nn.parallel import DistributedDataParallel as DDP

def get_name_from_filename(filename):
    name = ""
    for c in reversed(list(filename[:-3])):
        if c == "/":
            break

        name = c + name
        
    return name


def load_networks(rank, config):
    config["networks_path"]
    
def setup(rank, world_size, config, project_name):
    # Setup rendezvous
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    # Initalize WandB logging on rank 0
    if rank == 0 and project_name is not None:
        run = wandb.init(
                config  = config,
                entity  = "naddeok",
                project = project_name)
    else:
        run = None
        
    # Initialize the process group
    dist.init_process_group(backend    = "nccl",
                            init_method = "env://", 
                            rank        = rank, 
                            world_size  = world_size)
    
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    return run

def gather_variable(rank, world_size, variable):
    # Initialize lists
    variable_list = [variable.clone() for _ in range(world_size)]

    # Gather all variables
    dist.all_gather(variable_list, variable)
    
    # Convert from list to single tensor
    return torch.stack(variable_list)

def gather_metrics(rank, world_size, results, total_tested):
     # Gather values from all machines (ranks)
    results_list            = gather_variable(rank, world_size, results)
    total_tested_list       = gather_variable(rank, world_size, total_tested)
    dist.barrier()
    
    # Calculate final metric
    if rank == 0:
        metric = (total_tested_list * results_list).sum() / total_tested_list.sum()

        return metric.item()

    else:
        return None

def exp3(rank, world_size, config, project_name):

    ## Initialize WandB and Process Group
    run = setup(rank, world_size, config, project_name)
    
    ## Create models and move to GPUs with id rank
    list_of_networks, list_of_names = load_networks(rank, config)

    if True:
        # Attacker Network
        attacker_net = FstLayUniNet(set_name   = config["set_name"], 
                                    gpu        = rank,
                                    model_name = config["attack_model_name"]).to(rank)
        attacker_net_state_dict = torch.load(config["attack_net_filename"], map_location=torch.device('cpu'))
        if config["attack_net_from_ddp"]:  # Remove prefixes if from DDP
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(attacker_net_state_dict, "module.")
        attacker_net.load_state_dict(attacker_net_state_dict)
        attacker_net = DDP(attacker_net, device_ids=[rank])
        attacker_net.eval()

        # Regular Network
        reg_net = FstLayUniNet(set_name   = config["set_name"], 
                                gpu        = rank,
                                model_name = config["reg_model_name"]).to(rank)
        reg_net_state_dict = torch.load(config["reg_net_filename"], map_location=torch.device('cpu'))
        if config["reg_net_from_ddp"]:  # Remove prefixes if from DDP
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(reg_net_state_dict, "module.")
        reg_net.load_state_dict(reg_net_state_dict)
        reg_net = DDP(reg_net, device_ids=[rank])
        reg_net.eval()

        # Unitary Network
        U_net = FstLayUniNet(set_name   = config["set_name"], 
                                gpu        = rank,
                                U_filename = config["U_filename"],
                                model_name = config["U_model_name"]).to(rank)
        U_net_state_dict = torch.load(config["U_net_filename"], map_location=torch.device('cpu'))
        if config["U_net_from_ddp"]:  # Remove prefixes if from DDP
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(U_net_state_dict, "module.")
        U_net.load_state_dict(U_net_state_dict)
        U_net = DDP(U_net, device_ids=[rank])
        U_net.eval()

        # Distillation Network
        distill_net = FstLayUniNet(set_name   = config["set_name"], 
                                gpu        = rank,
                                model_name = config["distill_model_name"]).to(rank)
        distill_net_state_dict = torch.load(config["distill_net_filename"], map_location=torch.device('cpu'))
        if config["distill_net_from_ddp"]:  # Remove prefixes if from DDP
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(distill_net_state_dict, "module.")
        distill_net.load_state_dict(distill_net_state_dict)
        distill_net = DDP(distill_net, device_ids=[rank])
        distill_net.eval()

        # Adversarial Trained Network
        adv_pgd_net = FstLayUniNet(set_name   = config["set_name"], 
                                gpu        = rank,
                                model_name = config["adv_pgd_model_name"]).to(rank)
        adv_pgd_net_state_dict = torch.load(config["adv_pgd_net_filename"], map_location=torch.device('cpu'))
        if config["adv_pgd_net_from_ddp"]:  # Remove prefixes if from DDP
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(adv_pgd_net_state_dict, "module.")
        adv_pgd_net.load_state_dict(adv_pgd_net_state_dict)
        adv_pgd_net = DDP(adv_pgd_net, device_ids=[rank])
        adv_pgd_net.eval()

    ## Load Data
    data = Data(gpu          = rank, 
                set_name     = config["set_name"], 
                maxmin       = True,
                test_batch_size = config["batch_size"])

    # Test
    metrics = torch.zeros((len(list_of_networks), len(list_of_networks)))
    for i, net1 in enumerate(list_of_networks):
        for j, net2 in enumerate(list_of_networks):
            if i <= j:
                result, total_tested = comparison_metric(rank, net1, net2, data, config)
                metric = gather_metrics(rank, world_size, result, total_tested)

                if rank == 0:
                    metrics[i,j] = metric
    
    print("Adv Gathered", rank)
    # Log/Save data, results and model
    if rank == 0:
        
        # Save
        if config["save_to_excel"]:
            print("Saving ", config["metric_type"], " Manifold Comparison Results to Excel")
            from xlwt import Workbook  

            # Open Workbook
            wb = Workbook() 
            
            # Create sheet
            sheet = wb.add_sheet('Results') 
            
            # Make NSR column header
            sheet.write(0, 0, "NSR")

            # Add col and row headers
            for i, name in enumerate(list_of_names):
                # First col of headers
                sheet.write(i + 1, 0, name)

                # First row of headers
                sheet.write(0, i + 1, name)

            # Add data
            for i in metrics.size(0):
                for j in metrics.size(1):
                    if i <= j:
                        sheet.write(1 + i, 1 + j, metrics[i, j])
            
            # Save
            wb.save('results/' + config["set_name"]  + "/manifold_comparison/" + config["metric_type"] + ".xls") 
            print("Saved ", config["metric_type"], " Results")
        

    # Close all processes
    dist.barrier()
    dist.destroy_process_group()


def comparison_metric(rank, net1, net2, data, config):
    attacker1 = Attacker(net1, data, rank)
    attacker2 = Attacker(net2, data, rank)

    # Test data in test loader
    results = torch.zeros()
    total_tested = 0
    for inputs, labels in data.test_loader:
        # Push to gpu
        inputs, labels = inputs.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)

        if config["metric_type"] == "Max Eigenvector":
            max_eigenvector1, _ = attacker1.get_max_eigenpair(inputs, labels)
            max_eigenvector2, _ = attacker1.get_max_eigenpair(inputs, labels)
            
            for i in range(inputs.size(0)):
                results.append(np.abs(cos_sim(max_eigenvector1[i].view(1, -1), max_eigenvector2[i].view(1, -1)).item()))

            total_tested += inputs.size(0)

    results = torch.mean(torch.tensor(results).float()).item()
    return  results, total_tested


def run_ddp(func, world_size, config, project_name):
    
    # Spawn processes on gpus
    mp.spawn(func,
             args=(world_size, 
                    config,
                    project_name,),
             nprocs = world_size,
             join=True)

    # Display
    print("Run Complete")

if __name__ == "__main__":
    # Hyperparameters
    #-------------------------------------#
    # DDP
    gpu_ids = "1,2,3,4,5,6,7"
    
    # WandB
    project_name = None # "DDP MNIST"

    # Network
    config = {  
                # Networks
                # MNIST
                # "attack_model_name"   : "lenet", # "cifar10_mobilenetv2_x1_0"
                # "attack_net_filename" : "models/pretrained/MNIST/lenet_w_acc_97.pt", # "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt" # "models/pretrained/MNIST/lenet_w_acc_97.pt"
                # "attack_net_from_ddp" : True,
                # "attacker_net_acc"    : 0.97,

                # "reg_model_name"      : "lenet", # "cifar10_mobilenetv2_x1_0"
                # "reg_net_filename"    : "models/pretrained/MNIST/lenet_w_acc_98.pt",  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
                # "reg_net_from_ddp"    : True,
                # "reg_net_acc"         : 0.98,

                # "U_model_name"        : "lenet", # "cifar10_mobilenetv2_x1_4"
                # "U_net_filename"      :  "models/pretrained/MNIST/U_lenet_w_acc_94.pt",  # "models/pretrained/CIFAR10/U_cifar10_mobilenetv2_x1_4_w_acc_76.pt" # "models/pretrained/MNIST/Control_lenet_w_acc_97.pt" 
                # "U_filename"          : "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt", # "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt" # 
                # "U_net_from_ddp"      : True,
                # "U_net_acc"           : 0.94,

                # "distill_model_name"   : "lenet", # "cifar10_mobilenetv2_x1_0"
                # "distill_net_filename" : "models/pretrained/MNIST/distilled_20_lenet_w_acc_94.pt",  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
                # "distill_net_from_ddp" : True,
                # "distill_net_acc"      : 0.94,

                # "adv_pgd_model_name"   : "lenet", # "cifar10_mobilenetv2_x1_0"
                # "adv_pgd_net_filename" : "models/pretrained/MNIST/PGD_15_lenet_w_acc_97.pt",  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
                # "adv_pgd_net_from_ddp" : True,
                # "adv_pgd_net_acc"      : 0.97,

                # CIFAR10
                "attack_model_name"   : "cifar10_mobilenetv2_x1_0",
                "attack_net_filename" : "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt",
                "attack_net_from_ddp" : True,

                "reg_model_name"      : "cifar10_mobilenetv2_x1_0",
                "reg_net_filename"    : "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" ,
                "reg_net_from_ddp"    : True,

                "U_model_name"        : "cifar10_mobilenetv2_x1_4",
                "U_net_filename"      : "models/pretrained/CIFAR10/U_cifar10_mobilenetv2_x1_4_w_acc_76.pt",
                "U_filename"          : "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt",
                "U_net_from_ddp"      : True,

                "distill_model_name"   : "cifar10_mobilenetv2_x1_0",
                "distill_net_filename" : "models/pretrained/CIFAR10/distilled_20_cifar10_mobilenetv2_x1_0_w_acc_89.pt",
                "distill_net_from_ddp" : True,

                "adv_pgd_model_name"   : "cifar10_mobilenetv2_x1_0",
                "adv_pgd_net_filename" : "models/pretrained/CIFAR10/TEMP_PGD_15_cifar10_mobilenetv2_x1_0_w_acc_72.pt",
                "adv_pgd_net_from_ddp" : True,

                # Data
                "set_name"      : "CIFAR10",
                "batch_size"    : 64,

                # Criterion
                "crit"          : "cross_entropy",

                # Test Robustness
                "attack_type"                          : "Gaussian_Noise", # "CW2", # "PGD", #  "Gaussian Noise", # "FGSM", #
                "epsilons"                             : np.linspace(0, 1.0, num=101),
                "save_to_excel"                        : True
               }
    #-------------------------------------#

    # Set GPUs to Use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # Run training using DDP
    attack_types = ["Gaussian_Noise", "FGSM", "CW2"]
    for attack in attack_types:
        s = time()
        config["attack_type"] = attack
        run_ddp(exp2, n_gpus, config, project_name)
        print("Runtime ", time() - s)