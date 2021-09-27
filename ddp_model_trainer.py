import os
import wandb
import torch
import matplotlib
matplotlib.use('Agg')
import torch.distributed as dist
import torch.multiprocessing as mp
from adversarial_attacks import Attacker
from data_setup import Data
from models.classes.first_layer_unitary_net  import FstLayUniNet
from torch.nn.parallel import DistributedDataParallel as DDP

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

def initalize_scheduler(sched, optimizer, epochs):
    if sched == "Cosine Annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = int(epochs))
    else:
        print("Unknown Scheduler entered, please add to initalize_scheduler func.")
        exit()
    
    return scheduler

def initalize_criterion(criterion):
    # Setup Criterion
    if criterion=="cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    else: 
        print("Invalid criterion setting in sweep_config.py")
        exit()
        
    return criterion

def train(rank, world_size, config, project_name):

    # Initialize WandB and Process Group
    print(f"Training on rank {rank}.")
    run = setup(rank, world_size, config, project_name)

    # Create model and move it to GPU with id rank
    net = FstLayUniNet( set_name   = config["set_name"], 
                        gpu        = rank,
                        U_filename = config["U_filename"],
                        model_name = config["model_name"],
                        pretrained = config["pretrained"]).to(rank)

    net = DDP(net, device_ids=[rank])

    # Data loading code
    data = Data(gpu          = rank, 
                set_name     = config["set_name"], 
                data_augment = config["data_augment"],
                maxmin       = True if config["attack_type"] is not None else False)

    train_loader = data.get_train_loader(config["batch_size"])

    # Setup Optimzier and Criterion
    optimizer = initalize_optimizer(net, config)
    criterion = initalize_criterion(config["crit"])
    if config["sched"] is not None:
        scheduler = initalize_scheduler(config["sched"], optimizer, config["epochs"])

    correct = 0
    total_tested = 0
    total_step = len(train_loader)
    for epoch in range(int(config["epochs"])):    
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Push to gpu
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            # Adversarial Training
            if config["attack_type"] is not None and config["epoch_delay"] <= epoch:
                # Initalize Attacker with current model parameters
                attacker = Attacker(net = net, data =  data, gpu = rank)

                # Generate attacks and replace them with orginal images
                images = attacker.get_attack_accuracy(attack = config["attack_type"] ,
                                                        attack_images = images,
                                                        attack_labels = labels,
                                                        epsilons = [config["epsilon"]],
                                                        return_attacks_only = True,
                                                        prog_bar = False)

                dist.barrier()
                print("Attacks Generated on Rank ", rank)

            # Set the parameter gradients to zero
            optimizer.zero_grad()   

            #Forward pass
            with torch.set_grad_enabled(True):
                # SAM optimizer needs closure function to 
                #  reevaluate the loss function many times
                def closure():
                    #Set the parameter gradients to zero
                    optimizer.zero_grad()   

                    # Forward pass
                    outputs = net(images) 
                    
                    # Calculate loss
                    loss = criterion(outputs, orginal_labels)   

                    # Backward pass and optimize
                    loss.backward() 

                    return loss

                # Rerun forward pass and loss once SAM is done
                # Forward pass
                outputs = net(images) 

                # Calculate loss
                loss = criterion(outputs, labels) 

                _, predictions = torch.max(outputs, 1)
                correct += (predictions == labels).sum()
                total_tested += labels.size(0)
                
                epoch_loss += loss.item()
                
                # Update weights
                loss.backward() 
                if config["use_SAM"]:
                    optimizer.step(closure)
                else:
                    optimizer.step()

            # Display
            if (i + 1) % max(int(total_step/10), 2) == 0 and rank == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, config["epochs"], i + 1, total_step, loss.item()))

                # Log
                if run is not None:
                    run.log({"batch_loss": loss.item()})

        # Scheduler step
        if config["sched"] is not None:
            scheduler.step()

            if rank == 0:
                run.log({"sched lr": scheduler.get_last_lr()[0]})
            dist.barrier()

        # Display 
        if (epoch % 2 == 0) and (rank == 0):
            val_loss, val_acc = test(rank, net, data, config["crit"])

            # Late data augmentation
            # if ((val_loss > epoch_loss/len(data.trainset)) and (epoch > 10)) or (epoch > 0.9*config.epochs):
            #     data.data_augment = True
            #     data.train_set = data.get_trainset()
            #     train_loader = data.get_train_loader(config.batch_size)
            #     wandb.log({ "Data Augmentation" : data.data_augment})

            net.train(True)
            print("Epoch: ", epoch + 1, "\tTrain Loss: ", round(epoch_loss/len(data.train_set),5), "\tVal Loss: ", round(val_loss,5))

            run.log({   "epoch"      : epoch + 1, 
                        "Train Loss" : epoch_loss/len(data.train_set),
                        "Train Acc"  : correct/total_tested,
                        "Val Loss"   : val_loss,
                        "Val Acc"    : val_acc})
        dist.barrier()

    # End of training
    print("Done Training on Rank ", rank)

    # Test
    if rank == 0:
        print("Final Evaluation")
        val_loss, val_acc = test(rank, net, data, config["crit"])
        run.log({"epoch"      : epoch + 1, 
                "Train Loss"  : epoch_loss/len(data.train_set),
                "Train Acc"   : correct/total_tested,
                "Val Loss"    : val_loss,
                "Val Acc"     : val_acc})
        
        # Save Model
        if config["save_model"]:
            print("Saving Model")

            # Define File Names
            filename  = str(config["model_name"]) + "_w_acc_" + str(int(round(val_acc.item() * 100, 3))) + ".pt"
            if config["U_filename"] is not None:
                filename  = "U_" + filename
            
            if config["attack_type"] is not None:
                filename = config["attack_type"] + "_" + str(int(config["epsilon"]*100)) + "_" + filename
                
            
            # Save Models
            torch.save(net.state_dict(), "models/pretrained/" + config["set_name"]  + "/" + filename)

            
    # Wait till all processes are done thrn shut down
    dist.barrier()
    dist.destroy_process_group()

    # Close WandB
    if rank == 0 and run is not None:
        wandb.finish()
        print("WandB Finished")

def test(rank, net, data, crit):
    # Set to test mode
    net.eval()

    #Create loss functions
    criterion = initalize_criterion(crit)

    # Initialize
    total_loss = 0
    correct = 0
    total_tested = 0
    # Test data in test loader
    for images, labels in data.test_loader:
        # Push to gpu
        images, labels = images.to(rank, non_blocking=True), labels.to(rank, non_blocking=True)

        #Forward pass
        outputs = net(images)
        loss = criterion(outputs, labels) # Calculate loss 

        # Update runnin sum
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == labels).sum()
        total_tested += labels.size(0)
        total_loss   += loss.item()
    
    # Test Loss
    test_loss = (total_loss/len(data.test_set))
    test_acc  = correct / total_tested

    return test_loss, test_acc

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
    gpu_ids      = "0,1,2,3,4,5,6,7"

    # WandB
    project_name = "DDP MNIST"

    # Network
    config = {  
                # Network
                "model_name": "lenet",
                "pretrained": False,
                "save_model": True,

                # Data
                "set_name"      : "MNIST",
                "batch_size"    : 512,
                "data_augment"  : True,
                
                # Optimizer
                "optim"         : "sgd",
                "epochs"        : 25,
                "lr"            : 5e-1,
                "sched"         : "Cosine Annealing",
                "weight_decay"  : 1e-4,
                "momentum"      : 0.9,
                "use_SAM"       : False, 

                # Criterion
                "crit"          : "cross_entropy",

                # Hardening
                ### U Defense ###
                "U_filename"    : None, # "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt",

                ### Adv Train ###
                "attack_type"   : "PGD",
                "epsilon"       : 0.15,
                "epoch_delay"   : 5
                }
    #-------------------------------------#
    
    # Set GPUs to Use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # Run training using DDP
    run_ddp(train, n_gpus, config, project_name)

