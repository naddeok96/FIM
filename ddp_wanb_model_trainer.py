'''
This script will train a model and save it
'''
# Imports
import os
import copy
import wandb
import torch
import pickle
import random
from sam.sam import SAM
from data_setup import Data
import torch.multiprocessing as mp
from mnist_sweep_config import sweep_config
from models.classes.first_layer_unitary_net  import FstLayUniNet

# WandB Functions
#-------------------------------------------------------------------------------------#
def initalize_config_defaults(sweep_config):
    config_defaults = {}
    for key in sweep_config['parameters']:

        if list(sweep_config['parameters'][key].keys())[0] == 'values':
            config_defaults.update({key : sweep_config['parameters'][key]["values"][0]})

        else:
            config_defaults.update({key : sweep_config['parameters'][key]["min"]})

    wandb.init(config = config_defaults, group="DDP")

    return wandb.config

def initalize_net(set_name, rank, config):
    # Network 
    net = FstLayUniNet(set_name, gpu = rank,
                       U_filename = config.transformation,
                       model_name = config.model_name,
                       pretrained = config.pretrained)
    # net.load_state_dict(torch.load('models/pretrained/CIFAR10/Ucifar10_mobilenetv2_x1_4_w_acc_78.pt', map_location=torch.device('cpu')))

    # Return network
    return net.to(rank)

def initalize_optimizer(data, net, config):
    if config.optimizer=='sgd':
        if config.use_SAM:
            optimizer = SAM(net.parameters(), torch.optim.SGD,  lr=config.learning_rate,
                                        momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                        weight_decay=config.weight_decay)

    if config.optimizer=='nesterov':
        if config.use_SAM:
            optimizer = SAM(net.parameters(), torch.optim.SGD,  lr=config.learning_rate, momentum=config.momentum,
                                        weight_decay=config.weight_decay, nesterov=True)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=config.momentum,
                                    weight_decay=config.weight_decay, nesterov=True)

    if config.optimizer=='adam':
        if config.use_SAM:
            optimizer = SAM(net.parameters(), torch.optim.Adam,  lr=config.learning_rate, 
                                                weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    if config.optimizer=='adadelta':
        if config.use_SAM:
            optimizer = SAM(net.parameters(), torch.optim.Adadelta,  **{"lr" : config.learning_rate, 
                                                                 "weight_decay" : config.weight_decay, 
                                                                 "rho" : config.momentum})
        else:
            optimizer = torch.optim.Adadelta(net.parameters(), lr=config.learning_rate, 
                                        weight_decay=config.weight_decay, rho=config.momentum)


    return optimizer

def initalize_scheduler(optimizer, data, config):
    if config.scheduler == "Cosine Annealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = int(config.epochs))
    
    return scheduler

def initalize_criterion(config):
    # Setup Criterion
    if config.criterion=="mse":
        criterion = torch.nn.MSELoss()

    elif config.criterion=="cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()

    else: 
        print("Invalid criterion setting in sweep_config.py")
        exit()
        
    return criterion

def train(rank, world_size):
    save_model   = False
    data_augment = False
    set_name     = "MNIST"

    # Set ports
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Weights and Biases Setup
    config = initalize_config_defaults(sweep_config)
    wandb.log({ "Data Augmentation" : data_augment})

    # Get training data
    data = Data(gpu = rank, 
                set_name = set_name, 
                data_augment = data_augment)
                
    train_loader = data.get_train_loader(config.batch_size)

    # Initialize Network
    net = initalize_net(data.set_name, rank, config)
    net.train(True)

    # Setup Optimzier and Criterion
    optimizer = initalize_optimizer(data, net, config)
    criterion = initalize_criterion(config)
    if config.scheduler is not None:
        scheduler = initalize_scheduler(optimizer, data, config)
        
    # Loop for epochs
    correct = 0
    total_tested = 0
    for epoch in range(int(config.epochs)):    
        epoch_loss = 0
        for i, batch_data in enumerate(train_loader, 0): 
            # Get labels and inputs from train_loader
            inputs, labels = batch_data

            # One Hot the labels
            orginal_labels = copy.copy(labels).long()
            labels = torch.eq(labels.view(labels.size(0), 1), torch.arange(10).reshape(1, 10).repeat(labels.size(0), 1)).float()
                
            # Push to gpu
            orginal_labels = orginal_labels.to(rank)
            inputs, labels = inputs.to(rank), labels.to(rank)

            #Set the parameter gradients to zero
            optimizer.zero_grad()   

            #Forward pass
            with torch.set_grad_enabled(True):
                # SAM optimizer needs closure function to 
                #  reevaluate the loss function many times
                def closure():
                    #Set the parameter gradients to zero
                    optimizer.zero_grad()   

                    # Forward pass
                    outputs = net(inputs) 
                    
                    # Calculate loss
                    if config.criterion == "cross_entropy":
                        loss = criterion(outputs, orginal_labels) 
                    else:
                        loss = criterion(outputs, labels) # Calculate loss 

                    # Backward pass and optimize
                    loss.backward() 

                    return loss

                # Rerun forward pass and loss once SAM is done
                # Forward pass
                outputs = net(inputs) 

                # Calculate loss
                if config.criterion == "cross_entropy":
                    loss = criterion(outputs, orginal_labels) 
                else:
                    loss = criterion(outputs, labels) # Calculate loss 

                _, predictions = torch.max(outputs, 1)
                correct += (predictions == orginal_labels).sum()
                total_tested += labels.size(0)
                epoch_loss += loss.item()
                

                # Update weights
                loss.backward() 
                if config.use_SAM:
                    optimizer.step(closure)
                else:
                    optimizer.step()

        if config.scheduler is not None:
            scheduler.step()

        # Display 
        if (epoch % 2 == 0) and (rank == 0):
            # val_loss, val_acc = test(net, data, config)

            # if ((val_loss > epoch_loss/len(data.train_set)) and (epoch > 10)) or (epoch > 0.9*config.epochs):
            #     data.data_augment = True
            #     data.train_set = data.get_trainset()
            #     train_loader = data.get_train_loader(config.batch_size)
            #     wandb.log({ "Data Augmentation" : data.data_augment})

            # net.train(True)
            # print("Epoch: ", epoch + 1, "\tTrain Loss: ", epoch_loss/len(data.train_set), "\tVal Loss: ", val_loss)

            # wandb.log({ "epoch"      : epoch, 
            #             "Train Loss" : epoch_loss/len(data.train_set),
            #             "Train Acc"  : correct/total_tested,
            #             "Val Loss"   : val_loss,
            #             "Val Acc"    : val_acc})

            wandb.log({ "epoch"      : epoch, 
                        "Train Loss" : epoch_loss/len(data.train_set),
                        "Train Acc"  : correct/total_tested})

    # # Test
    # val_loss, val_acc = test(net, data, config)
    # wandb.log({"epoch"        : epoch, 
    #             "Train Loss"  : epoch_loss/len(data.train_set),
    #             "Train Acc"   : correct/total_tested,
    #             "Val Loss"    : val_loss,
    #             "Val Acc"     : val_acc})

    # # Save Model
    # if save_model:
    #     # Define File Names
    #     if net.U is not None:
    #         filename  = "U" + str(config.model_name) + "_w_acc_" + str(int(round(val_acc.item() * 100, 3))) + ".pt"
    #     else:
    #         filename  = "U" + str(config.model_name) + "_w_acc_" + str(int(round(val_acc.item() * 100, 3))) + ".pt"
        
    #     # Save Models
    #     torch.save(net.state_dict(), "models/pretrained/" + set_name  + "/" + filename)

    #     # Save U
    #     if net.U is not None:
    #         torch.save(net.U, "models/pretrained/" + set_name  + "/" + str(config.transformation) + "_for_" + set_name + filename)

    # Shut down all groups
    torch.distributed.destroy_process_group()
    
def test(net, data, config):
    # Set to test mode
    net.eval()

    #Create loss functions
    criterion = initalize_criterion(config)

    # Initialize
    total_loss = 0
    correct = 0
    total_tested = 0
    # Test data in test loader
    for i, batch_data in enumerate(data.test_loader, 0):
        # Get labels and inputs from train_loader
        inputs, labels = batch_data

        # One Hot the labels
        orginal_labels = copy.copy(labels).long()
        labels = torch.eq(labels.view(labels.size(0), 1), torch.arange(10).reshape(1, 10).repeat(labels.size(0), 1)).float()

        # Push to gpu
        if gpu:
            orginal_labels = orginal_labels.cuda()
            inputs, labels = inputs.cuda(), labels.cuda()

        #Forward pass
        outputs = net(inputs)
        if config.criterion == "cross_entropy":
            loss = criterion(outputs, orginal_labels) # Calculate loss
        else:
            loss = criterion(outputs, labels) # Calculate loss 

        # Update runnin sum
        _, predictions = torch.max(outputs, 1)
        correct += (predictions == orginal_labels).sum()
        total_tested += labels.size(0)
        total_loss   += loss.item()
    
    # Test Loss
    test_loss = (total_loss/len(data.test_set))
    test_acc  = correct / total_tested

    return test_loss, test_acc
#-------------------------------------------------------------------------------------#
   
def ddp_sweep():
    # Setup DDP WandB Sweep
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    mp.spawn(train,
            args=(n_gpus,),
            nprocs=n_gpus,
            join=True)
    
# Main
if __name__ == "__main__":
    # Hyperparameters
    gpu_ids      = "4, 5"
    project_name = "DDP MNIST"
    os.environ['WANDB_MODE'] = 'dryrun'

    # Set GPUs to Use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    # WandB Sweep
    sweep_id = wandb.sweep(sweep_config, entity="naddeok", project=project_name)
    wandb.agent(sweep_id, function=ddp_sweep)

    
    

