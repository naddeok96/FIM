import os
import wandb
import torch
import matplotlib
matplotlib.use('Agg')
from copy import copy
import numpy as np
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
    os.environ['MASTER_PORT'] = '12355'

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
    return run

def initalize_optimizer(net, config):
    if config["optim"] =='sgd':
        if config["use_SAM"]:
            optimizer = SAM(net.parameters(), torch.optim.SGD,  lr=config["lr"],
                                        momentum=config["momentum"], weight_decay=config["weight_decay"])
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"],
                                        weight_decay=config["weight_decay"])

    elif config["optim"]=='nesterov':
        if config["use_SAM"]:
            optimizer = SAM(net.parameters(), torch.optim.SGD,  lr=config["lr"], momentum=config["momentum"],
                                        weight_decay=config["weight_decay"], nesterov=True)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=config["lr"], momentum=config["momentum"],
                                    weight_decay=config["weight_decay"], nesterov=True)

    elif config["optim"]=='adam':
        if config["use_SAM"]:
            optimizer = SAM(net.parameters(), torch.optim.Adam,  lr=config["lr"], 
                                                weight_decay=config["weight_decay"])
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    elif config["optim"]=='adadelta':
        if config["use_SAM"]:
            optimizer = SAM(net.parameters(), torch.optim.Adadelta,  **{"lr" : config["lr"], 
                                                                 "weight_decay" : config["weight_decay"], 
                                                                 "rho" : config["momentum"]})
        else:
            optimizer = torch.optim.Adadelta(net.parameters(), lr=config["lr"], 
                                        weight_decay=config["weight_decay"], rho=config["momentum"])

    else:
        print("Unknown optimizer entered, please add to initalize_optimizer func.")
        exit()

    return optimizer

def initalize_scheduler(optimizer, config, steps_per_epoch = None):
    if config["sched"] == "Cosine Annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = int(config["epochs"]))

    elif config["sched"] == "One Cycle LR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr = config["lr"], 
                                                        epochs = config["epochs"],
                                                        steps_per_epoch = steps_per_epoch)
    else:
        print("Unknown Scheduler entered, please add to initalize_scheduler func.")
        exit()
    
    return scheduler

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

def train(rank, world_size, config, project_name):

    # Initialize WandB and Process Group
    run = setup(rank, world_size, config, project_name)

    # Create model and move it to GPU with id rank
    if config["pretrained_weights_filename"] or config["distill"]:
        state_dict = torch.load(config["pretrained_weights_filename"], map_location=torch.device('cpu'))

        if config["from_ddp"]:  # Remove prefixes if from DDP
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")


    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    net = FstLayUniNet( set_name   = config["set_name"], 
                        gpu        = rank,
                        U_filename = config["U_filename"],
                        model_name = config["model_name"]).to(rank)
    
    if config["distill"]: # Load teacher net
        teacher_net = FstLayUniNet( set_name   = config["set_name"], 
                                    gpu        = rank,
                                    U_filename = config["U_filename"],
                                    model_name = config["model_name"]).to(rank)
        teacher_net.load_state_dict(state_dict)
        teacher_net = DDP(teacher_net, device_ids=[rank])
        teacher_net.eval()

    else:
        if config["pretrained_weights_filename"] is not None:
            net.load_state_dict(state_dict)

    net = DDP(net, device_ids=[rank])

    # Load Data
    data = Data(gpu          = rank, 
                set_name     = config["set_name"], 
                data_augment = config["data_augment"],
                maxmin       = True,
                test_batch_size = config["batch_size"])
    train_loader = data.get_train_loader(config["batch_size"])

    # Setup Optimzier and Criterion
    optimizer = initalize_optimizer(net, config)
    criterion = initalize_criterion(config)
    if config["sched"] is not None:
        scheduler = initalize_scheduler(optimizer, config, steps_per_epoch = len(train_loader))

    # Disable adversarial robustness test while trinaing
    test_robustness = copy(config["test_robustness"])
    config["test_robustness"] = False

    # Begin Training
    if config["epochs"] != 0:
        print("Trainning on Rank", rank)
    for epoch in range(int(config["epochs"])):   
        train_loader.sampler.set_epoch(epoch)
        epoch_correct, epoch_total_tested, epoch_total_loss = [torch.tensor(0).float().to(rank) for _ in range(3)]
        for i, (inputs, labels) in enumerate(train_loader):
            # Push to gpu
            inputs = inputs.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            # Adversarial Training
            if config["attack_type"] is not None and config["epoch_delay"] <= epoch:
                # Initalize Attacker with current model parameters
                attacker = Attacker(net = net, data =  data, gpu = rank)

                # Generate attacks and replace them with orginal inputs
                inputs = attacker.get_attack_accuracy(attack = config["attack_type"] ,
                                                        attack_images = inputs,
                                                        attack_labels = labels,
                                                        epsilons = [config["epsilon"]],
                                                        return_attacks_only = True,
                                                        prog_bar = False)

                dist.barrier()

            # Set the parameter gradients to zero
            optimizer.zero_grad()   

            #Forward pass
            with torch.set_grad_enabled(True):
                # SAM optimizer needs closure function to reevaluate the loss function many times
                def closure():
                    #Set the parameter gradients to zero
                    optimizer.zero_grad()   

                    # Forward pass
                    outputs = net(inputs) 
                    
                    # Calculate loss
                    loss = criterion(outputs, orginal_labels)   

                    # Backward pass and optimize
                    loss.backward() 

                    if config["gradient clip"]:
                        torch.nn.utils.clip_grad_value_(net.parameters(), config["gradient clip"])

                    return loss

                # Rerun forward pass and loss once SAM is done
                # Forward pass
                outputs = net(inputs) 

                # Get loss
                if config["distill"]:
                    ## Sanity check that this method is equivalent to oringal criterion
                    # batch_size = labels.size(0)
                    # label_onehot = torch.FloatTensor(batch_size, data.num_classes)
                    # label_onehot.zero_()
                    # label_onehot.scatter_(1, labels.view(-1, 1), 1)
                    # print("One Hot", label_onehot[0])
                    # print(torch.sum(-label_onehot * F.log_softmax(outputs, -1), -1).mean())
                    
                    soft_labels = F.softmax(teacher_net(inputs) / config["distill_temp"], -1)
                    loss = torch.sum(-soft_labels * F.log_softmax(outputs, -1), -1).mean()

                else:
                    loss = criterion(outputs, labels)

                    if config["gradient clip"]:
                        torch.nn.utils.clip_grad_value_(net.parameters(), config["gradient clip"])

                _, predictions      = torch.max(outputs, 1)
                epoch_correct      += (predictions == labels).sum()
                epoch_total_tested += labels.size(0)
                epoch_total_loss   += loss.item()
                
                # Update weights
                loss.backward() 
                if config["use_SAM"]:
                    optimizer.step(closure)
                else:
                    optimizer.step()

            # Scheduler step
            if config["sched"] == "One Cycle LR":
                scheduler.step()

        # Scheduler step
        if config["sched"] == "Cosine Annealing":
            scheduler.step()

            if rank == 0:
                run.log({"sched lr": scheduler.get_last_lr()[0]})
            dist.barrier()

        # Display 
        if ((epoch + 1) % 10 == 0):
            # Test
            val_correct, val_total_tested, val_total_loss, _ = test(rank, net, data, config)
            net.train(True)

            epoch_acc, epoch_loss = gather_acc_and_loss(rank,  world_size, epoch_correct, epoch_total_tested, epoch_total_loss)
            val_acc, val_loss     = gather_acc_and_loss(rank,  world_size,   val_correct,   val_total_tested,   val_total_loss)
            
            if rank == 0:
                # Late data augmentation
                # if ((val_loss > epoch_loss/len(data.trainset)) and (epoch > 10)) or (epoch > 0.9*config.epochs):
                #     data.data_augment = True
                #     data.train_set = data.get_trainset()
                #     train_loader = data.get_train_loader(config.batch_size)
                #     wandb.log({ "Data Augmentation" : data.data_augment})

                
                print(  "Epoch: ",        epoch + 1, 
                        "\tTrain Loss: ", round(epoch_loss.item(),5), 
                        "\tVal Loss: ",   round(val_loss.item(),5))

                run.log({   "epoch"  : epoch + 1, 
                        "Train Loss" : epoch_loss,
                        "Train Acc"  : epoch_acc,
                        "Val Loss"   : val_loss,
                        "Val Acc"    : val_acc})

            dist.barrier()
   
    # Test
    print("Testing on Rank", rank)
    config["test_robustness"] = copy(test_robustness)
    val_correct, val_total_tested, val_total_loss, val_adv_correct = test(rank, net, data, config)
    
    # Gather values from all machines (ranks)
    print("Evaluating Results", rank)
    epoch_acc, epoch_loss = gather_acc_and_loss(rank,  world_size, epoch_correct, epoch_total_tested, epoch_total_loss) if config["epochs"] != 0 else None, None
    val_acc, val_loss     = gather_acc_and_loss(rank,  world_size,   val_correct,   val_total_tested,   val_total_loss)
    print("Val Gathered", rank)

    # Get adversarial accuracy 
    if config["test_robustness"]:

        fool_ratio = [None for _ in range(len(val_adv_correct))]
        for i, (epsilon, adv_correct) in enumerate(zip(config["attacker epsilons"], val_adv_correct)):
            adv_acc, _    = gather_acc_and_loss(rank,  world_size,   adv_correct,   val_total_tested,   torch.tensor(0).to(rank))
            if rank == 0:
                fool_ratio[i] = torch.round(100*((val_acc - adv_acc) / val_acc))

            dist.barrier()
    else:
        fool_ratio = None
    print("Adv Gathered", rank)
    # Log/Save data, results and model
    if rank == 0:
        
        print("Logging")     
        run.log({"Epoch"      : epoch + 1 if config["epochs"] != 0 else None, 
                "Train Loss"  : epoch_loss,
                "Train Acc"   : epoch_acc,
                "Val Loss"    : val_loss,
                "Val Acc"     : val_acc})

        print("Log")
        if config["test_robustness"]:
            fool_ratio_table = wandb.Table( columns = ["NSR", "Fool Ratio"],
                                        data    = [[eps, int(fool.item())] for eps, fool in zip(config["attacker epsilons"], fool_ratio)])
            
            run.log({"Fool Ratios" : fool_ratio_table})
            print("Table")

        if config["save_attack_results"] and config["test_robustness"]:
            print("Saving Attack Results to Excel")
            from xlwt import Workbook  

            # Open Workbook
            wb = Workbook() 
            
            # Create sheet
            sheet = wb.add_sheet('Results') 
            
            sheet.write(0, 0, "NSR")
            sheet.write(0, 1, "Fool Ratio")
            for i, (eps, fool) in enumerate(zip(config["attacker epsilons"], fool_ratio)):
                    sheet.write(i + 1, 0, eps)
                    sheet.write(i + 1, 1, int(fool.item()))
            
            attacker_name = get_name_from_filename(config["attacker_pretrained_weights_filename"])
            target_name   = get_name_from_filename(config["pretrained_weights_filename"])

            filename = config["attacker_attack_type"] + "_from_" + attacker_name + '_on_' + target_name + '_attack_results.xls'
            wb.save('results/' + data.set_name + '/' + filename) 
        print("Save Attack Results")
        if config["save_model"]:
            print("Saving Model")

            # Define File Names
            filename  = str(config["model_name"]) + "_w_acc_" + str(int(round(val_acc.item() * 100, 3))) + ".pt"
            if config["U_filename"] is not None:
                filename  = "U_" + filename
            
            if config["attack_type"] is not None:
                filename = config["attack_type"] + "_" + str(int(config["epsilon"]*100)) + "_" + filename

            if config["distill"]:
                filename = "distilled_" + str(config["distill_temp"]) + "_" + filename
                
            # Save Models
            torch.save(net.state_dict(), "models/pretrained/" + config["set_name"]  + "/" + filename)
        print("Model Saved")
    # Close all processes
    dist.barrier()
    dist.destroy_process_group()

    # Close WandB
    if rank == 0 and run is not None:
        wandb.finish()
        print("WandB Finished")

def test(rank, net, data, config):
    # Set to test mode
    net.eval()

    # Load Attacker 
    if config["test_robustness"]:
        attacker_net = FstLayUniNet( set_name  = config["set_name"], 
                                    gpu        = rank,
                                    model_name = config["model_name"]).to(rank)
        attacker_state_dict = torch.load(config["attacker_pretrained_weights_filename"], map_location=torch.device('cpu'))

        if config["from_ddp"]:  # Remove prefixes if from DDP
            torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(attacker_state_dict, "module.")
        attacker_net.load_state_dict(attacker_state_dict)
        attacker_net = DDP(attacker_net, device_ids=[rank])
        attacker_net.eval()

        # Initalize Attacker with current model parameters
        attacker = Attacker(net = attacker_net, data = data, gpu = rank)

    #Create loss functions
    criterion = initalize_criterion(config)
    indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')

    # Initialize
    correct, total_tested, total_loss   = [torch.tensor(0).float().to(rank) for _ in range(3)]
    adv_correct  = [torch.tensor(0).float().to(rank) if config["test_robustness"] else None for _ in range(len(config["attacker epsilons"]))]
    # Test data in test loader
    for inputs, labels in data.test_loader:
        # Push to gpu
        inputs, labels = inputs.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)

        #Forward pass
        outputs = net(inputs)

        # Test on adversarial attacks
        if config["test_robustness"]:
            # Declare indivdual cirterion to ensure attack increases loss
            losses  = indv_criterion(outputs, labels)

            # Generate attacks and replace them with orginal inputs
            normed_perturbations = attacker.get_attack_accuracy(attack                    = config["attacker_attack_type"] ,
                                                                attack_images             = inputs,
                                                                attack_labels             = labels,
                                                                return_perturbations_only = True,
                                                                prog_bar                  = False)

            # Cycle over all espiplons
            for i, epsilon in enumerate(config["attacker epsilons"]):
                
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
             nprocs=world_size,
             join=True)

    # Display
    print("Run Complete")

if __name__ == "__main__":
    # Hyperparameters
    #-------------------------------------#
    # DDP
    gpu_ids = "0, 1, 2, 3, 4, 5, 6, 7"

    # WandB
    project_name = "DDP CIFAR10"

    # Network
    config = {  
                # Network
                "model_name"                  : "cifar10_mobilenetv2_x1_0",
                "pretrained_weights_filename" : "models/pretrained/CIFAR10/Nonecifar10_mobilenetv2_x1_0_w_acc_91.pt",
                "from_ddp"                    : False,
                "save_model"                  : True,

                # Attacker Network
                "test_robustness"                       : False,
                "save_attack_results"                   : False,
                "attacker_pretrained_weights_filename"  : "models/pretrained/MNIST/lenet_w_acc_98.pt",
                "attacker_attack_type"                  : "OSSA", # "CW2", # "PGD", #  "Gaussian Noise", # "FGSM", # 
                "attacker epsilons"                     : np.linspace(0, 1.0, num=101),

                # Data
                "set_name"      : "CIFAR10",
                "batch_size"    : 400,
                "data_augment"  : True,
                
                # Optimizer
                "optim"         : "adam",
                "epochs"        : 500,
                "lr"            : 0.01,
                "sched"         : "One Cycle LR", #"Cosine Annealing", # 
                "gradient clip" : 0.1,
                "weight_decay"  : 1e-4,
                "momentum"      : 0.9,
                "use_SAM"       : False, 

                # Criterion
                "crit"          : "cross_entropy",

                # Hardening
                ### Unitary ###
                "U_filename"    : None, # "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt", # "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt",

                ### Adv Train ###
                "attack_type"   : "PGD",
                "epsilon"       : 0.15,
                "epoch_delay"   : 0,

                ### Distill ###
                "distill"       : False,
                "distill_temp"  : 20,
               }
    #-------------------------------------#

    # Set GPUs to Use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # Run training using DDP
    run_ddp(train, n_gpus, config, project_name)

    # target_networks = [ "models/pretrained/MNIST/distilled_20_lenet_w_acc_94.pt",
    #                     "models/pretrained/MNIST/PGD_15_lenet_w_acc_97.pt",
    #                     "models/pretrained/MNIST/U_lenet_w_acc_94.pt",
    #                     "models/pretrained/MNIST/lenet_w_acc_98.pt"]
    # attacker_attack_types = ["Gaussian_Noise", "FGSM", "PGD",  "CW2", "OSSA"]
    # for attacker_attack_type in attacker_attack_types:
        # print("Working on", attacker_attack_type)
        # config["attacker_attack_type"] = attacker_attack_type

        # for target_network in target_networks:
        #     print("\tWorking on", get_name_from_filename(target_network))
        #     config["pretrained_weights_filename"] = target_network

        #     if target_network == "models/pretrained/MNIST/U_lenet_w_acc_94.pt":
        #         config["U_filename"] = "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt"
        #     else:
        #         config["U_filename"] = None
        #     run_ddp(train, n_gpus, config, project_name)

