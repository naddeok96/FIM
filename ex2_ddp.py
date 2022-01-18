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

def initalize_criterion(config):
    # Setup Criterion
    if config["crit"] == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    else: 
        print("Invalid criterion setting in sweep_config.py")
        exit()
        
    return criterion

def gather_variable(rank, world_size, variable):
    # Initialize lists
    variable_list = [variable.clone() for _ in range(world_size)]

    # Gather all variables
    dist.all_gather(variable_list, variable)
    
    # Convert from list to single tensor
    return torch.stack(variable_list)

def gather_acc_and_loss(rank,  world_size, correct, total_tested, total_loss):    
    # Gather values from all machines (ranks)
    correct_list            = gather_variable(rank, world_size, correct)
    total_tested_list       = gather_variable(rank, world_size, total_tested)
    total_loss_list         = gather_variable(rank, world_size, total_loss)
    dist.barrier()
    
    # Calculate final metrics
    if rank == 0:
        acc = correct_list.sum() / total_tested_list.sum()
        loss = (total_tested_list * total_loss_list).sum() / total_tested_list.sum()

        return acc, loss

    else:
        return None, None

def exp2(rank, world_size, config, project_name):

    ## Initialize WandB and Process Group
    run = setup(rank, world_size, config, project_name)
    
    ## Create models and move to GPUs with id rank
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

    # Get regular attack accuracies on attacker network
    # print("Working on White Box Attacks...")
    # white_box_accs  = attacker.get_attack_accuracy(attack = attack_type, epsilons = epsilons)                                          
    # white_box_fool_ratio = attacker.get_fool_ratio(attacker_net_acc, white_box_accs)
    # table.add_column("White Box Fool Ratio", white_box_fool_ratio)
    # results.append(white_box_fool_ratio)
    # names.append("White Box Attack")

    # Test
    results = []
    for net in [attacker_net, reg_net, U_net, distill_net, adv_pgd_net]:
        val_correct, val_total_tested, val_total_loss, val_adv_correct = test(rank, net, attacker_net, data, config)
        val_acc, val_loss     = gather_acc_and_loss(rank,  world_size,   val_correct,   val_total_tested,   val_total_loss)

        fool_ratio = [None for _ in range(len(val_adv_correct))]
        for i, (epsilon, adv_correct) in enumerate(zip(config["epsilons"], val_adv_correct)):
            adv_acc, _    = gather_acc_and_loss(rank,  world_size,   adv_correct,   val_total_tested,   torch.tensor(0).to(rank))
            if rank == 0:
                fool_ratio[i] = torch.round(100*((val_acc - adv_acc) / val_acc)).item()

            dist.barrier()
        
        if rank == 0:
            results.append(fool_ratio)

    
    print("Adv Gathered", rank)
    # Log/Save data, results and model
    if rank == 0:
        # Display
        table = PrettyTable()
        table.add_column("Epsilons", config["epsilons"])
        headers = ["White Box", "Reg Net", "U Net", "Distill Net", "AT-PGD Net"]
        for i in range(len(results)):
            table.add_column(headers[i], results[i])
            plt.plot(config["epsilons"], results[i], label = headers[i])


        print(table)
        plt.title(config["attack_type"] + " on " + config["set_name"])
        plt.legend()
        plt.savefig('results/' + config["set_name"]  + "/" + config["attack_type"] + "/defense_comparison.png")

        # Save
        if config["save_to_excel"]:
            print("Saving Attack Results to Excel")
            from xlwt import Workbook  

            # Open Workbook
            wb = Workbook() 
            
            # Create sheet
            sheet = wb.add_sheet('Results') 
            
            # Make NSR column header
            sheet.write(0, 0, "NSR")
            
            for i in range(len(results)):
                # Other column headers
                sheet.write(0, i + 1, headers[i])

                # Populate rows
                fool_ratio = results[i]
                for j, (eps, fool) in enumerate(zip(config["epsilons"], fool_ratio)):
                        # Populate NSR column with first network
                        if i == 0:
                            sheet.write(j + 1, 0, eps)

                        # Populate other networks columns
                        sheet.write(j + 1, i + 1, int(fool))
                
            # Save
            wb.save('results/' + config["set_name"]  + "/" + config["attack_type"] + "/defense_comparison.png.xls") 
            print("Save Attack Results")
        

    # Close all processes
    dist.barrier()
    dist.destroy_process_group()

    # Close WandB
    if rank == 0 and run:
        wandb.finish()
        print("WandB Finished")

def test(rank, net, attacker_net, data, config):
    #Create loss functions
    criterion      = initalize_criterion(config)
    indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')

    # Initalize Attacker with current model parameters
    attacker = Attacker(net = attacker_net, data = data, gpu = rank)

    # Initialize
    correct, total_tested, total_loss   = [torch.tensor(0).float().to(rank) for _ in range(3)]
    adv_correct  = [torch.tensor(0).float().to(rank) for _ in range(len(config["epsilons"]))]
    # Test data in test loader
    for inputs, labels in data.test_loader:
        # Push to gpu
        inputs, labels = inputs.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)

        #Forward pass
        outputs = net(inputs)
        losses  = indv_criterion(outputs, labels)

        # Generate attacks and replace them with orginal inputs
        normed_perturbations = attacker.get_attack_accuracy(attack                    = config["attack_type"] ,
                                                            attack_images             = inputs,
                                                            attack_labels             = labels,
                                                            transfer_network          = net,
                                                            return_perturbations_only = True,
                                                            prog_bar                  = False)

        # Cycle over all espiplons
        for i, epsilon in enumerate(config["epsilons"]):
            
            # Set the unit norm of the highest eigenvector to epsilon
            input_norms = torch.linalg.norm(inputs.view(inputs.size(0), 1, -1), ord=None, dim=2).view(-1, 1, 1)
            perturbations = float(epsilon) * input_norms * normed_perturbations

            # Declare attacks as the perturbation added to the image                    
            attacks = (inputs.view(inputs.size(0), 1, -1) + perturbations).view(inputs.size(0), data.num_channels, data.image_size, data.image_size)

            # Check if loss has increased
            adv_outputs = net(attacks)
            adv_losses  = indv_criterion(adv_outputs, labels)

            # If losses has not increased flip direction
            signs = (losses < adv_losses).type(torch.float) 
            signs[signs == 0] = -1
            perturbations = signs.view(-1, 1, 1) * perturbations
        
            # Compute attack and models prediction of it
            attacks = (inputs.view(inputs.size(0), 1, -1) + perturbations).view(inputs.size(0), data.num_channels, data.image_size, data.image_size)

            # Feedforward
            adv_outputs = net(attacks)

            # Update running sum
            _, adv_predictions = torch.max(adv_outputs, 1)
            adv_correct[i] += (adv_predictions == labels).sum()

        # Examine output
        loss           = criterion(outputs, labels) 
        _, predictions = torch.max(outputs, 1)

        # Update running sum
        correct      += (predictions == labels).sum()
        total_tested += labels.size(0)
        total_loss   += loss.item()

    dist.barrier()
    return correct, total_tested, total_loss, adv_correct

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
    gpu_ids = "1,2,4"
    
    # WandB
    project_name = None # "DDP MNIST"

    # Network
    config = {  
                # Networks
                # MNIST
                "attack_model_name"   : "lenet", # "cifar10_mobilenetv2_x1_0"
                "attack_net_filename" : "models/pretrained/MNIST/lenet_w_acc_97.pt", # "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt" # "models/pretrained/MNIST/lenet_w_acc_97.pt"
                "attack_net_from_ddp" : True,
                "attacker_net_acc"    : 0.97,

                "reg_model_name"      : "lenet", # "cifar10_mobilenetv2_x1_0"
                "reg_net_filename"    : "models/pretrained/MNIST/lenet_w_acc_98.pt",  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
                "reg_net_from_ddp"    : True,
                "reg_net_acc"         : 0.98,

                "U_model_name"        : "lenet", # "cifar10_mobilenetv2_x1_4"
                "U_net_filename"      :  "models/pretrained/MNIST/U_lenet_w_acc_94.pt",  # "models/pretrained/CIFAR10/U_cifar10_mobilenetv2_x1_4_w_acc_76.pt" # "models/pretrained/MNIST/Control_lenet_w_acc_97.pt" 
                "U_filename"          : "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt", # "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt" # 
                "U_net_from_ddp"      : True,
                "U_net_acc"           : 0.94,

                "distill_model_name"   : "lenet", # "cifar10_mobilenetv2_x1_0"
                "distill_net_filename" : "models/pretrained/MNIST/distilled_20_lenet_w_acc_94.pt",  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
                "distill_net_from_ddp" : True,
                "distill_net_acc"      : 0.94,

                "adv_pgd_model_name"   : "lenet", # "cifar10_mobilenetv2_x1_0"
                "adv_pgd_net_filename" : "models/pretrained/MNIST/PGD_15_lenet_w_acc_97.pt",  # "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt" 
                "adv_pgd_net_from_ddp" : True,
                "adv_pgd_net_acc"      : 0.97,

                # CIFAR10
                # "attack_model_name"   : "cifar10_mobilenetv2_x1_0",
                # "attack_net_filename" : "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt",
                # "attack_net_from_ddp" : True,

                # "reg_model_name"      : "cifar10_mobilenetv2_x1_0",
                # "reg_net_filename"    : "models/pretrained/CIFAR10/cifar10_mobilenetv2_x1_0_w_acc_93.pt" ,
                # "reg_net_from_ddp"    : True,

                # "U_model_name"        : "cifar10_mobilenetv2_x1_4",
                # "U_net_filename"      : "models/pretrained/CIFAR10/U_cifar10_mobilenetv2_x1_4_w_acc_76.pt",
                # "U_filename"          : "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt",
                # "U_net_from_ddp"      : True,

                # "distill_model_name"   : "cifar10_mobilenetv2_x1_0",
                # "distill_net_filename" : "models/pretrained/CIFAR10/distilled_20_cifar10_mobilenetv2_x1_0_w_acc_89.pt",
                # "distill_net_from_ddp" : True,

                # "adv_pgd_model_name"   : "cifar10_mobilenetv2_x1_0",
                # "adv_pgd_net_filename" : "models/pretrained/CIFAR10/TEMP_PGD_15_cifar10_mobilenetv2_x1_0_w_acc_72.pt",
                # "adv_pgd_net_from_ddp" : True,

                # Data
                "set_name"      : "MNIST",
                "batch_size"    : 512,

                # Criterion
                "crit"          : "cross_entropy",

                # Test Robustness
                "attack_type"                          : "Gaussian_Noise", # "CW2", # "PGD", #  "Gaussian Noise", # "FGSM", #
                "epsilons"                             : np.linspace(0, 1.0, num=201),
                "save_to_excel"                        : True
               }
    #-------------------------------------#

    # Set GPUs to Use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # Run training using DDP
    attack_types = ["CW2"]
    for attack in attack_types:
        s = time()
        config["attack_type"] = attack
        run_ddp(exp2, n_gpus, config, project_name)
        print("Runtime ", time() - s)