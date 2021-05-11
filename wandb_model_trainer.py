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
from sam import SAM
from data_setup import Data
# from effnet_sweep_config import sweep_config
from dimah_net_sweep_config import sweep_config
from models.classes.first_layer_unitary_effnet    import FstLayUniEffNet
from models.classes.first_layer_unitary_dimah_net import FstLayUniDimahNet

# Functions
#-------------------------------------------------------------------------------------#
def initalize_config_defaults(sweep_config):
    config_defaults = {}
    for key in sweep_config['parameters']:
        config_defaults.update({key : random.choice(sweep_config['parameters'][key]["values"])})

    return config_defaults

def initalize_net(set_name, gpu, config):
    # Network'
    net = FstLayUniDimahNet(gpu = gpu)
    # net = FstLayUniEffNet(set_name = set_name,
    #                       gpu = gpu,
    #                       model_name = config.model_name,
    #                       pretrained = config.pretrained,
    #                       desired_image_size = 224)
    net = net.cuda() if gpu == True else net
    # Add unitary transformation
    if config.transformation == "R":
        net.set_random_matrix()

    elif config.transformation == "U":
        net.set_orthogonal_matrix()

    elif isinstance(config.transformation, str):
        with open(config.transformation, 'rb') as input:
            net.U = pickle.load(input).type(torch.FloatTensor)

    # Return network
    return net

def initalize_optimizer(net, config):
    if config.optimizer=='sgd':
        optimizer = torch.optim.SGD(net.parameters(),lr=config.learning_rate, momentum=config.momentum,
        weight_decay=config.weight_decay)

    if config.optimizer=="adadelta":
        optimizer = torch.optim.Adadelta(net.parameters(), lr=config.learning_rate, 
                                        weight_decay=config.weight_decay, rho=config.momentum)
    if config.use_SAM:
        optimizer = SAM(net.parameters(), optimizer)

    return optimizer

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

def train(data, save_model):
    # Weights and Biases Setup
    config_defaults = initalize_config_defaults(sweep_config)
    wandb.init(config = config_defaults)
    config = wandb.config

    #Get training data
    train_loader = data.get_train_loader(config.batch_size)

    # Initialize Network
    net = initalize_net(data.set_name, data.gpu, config)
    net.train(True)

    # Save Model
    if save_model:
        wandb.watch(net)

    # Setup Optimzier and Criterion
    optimizer = initalize_optimizer(net, config)
    criterion = initalize_criterion(config)
        
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
            if data.gpu:
                orginal_labels = orginal_labels.cuda()
                inputs, labels = inputs.cuda(), labels.cuda()

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
                if config.use_SAM:
                    optimizer.step(closure)
                else:
                    optimizer.step()

        # Display 
        if epoch % 10 == 0:
            val_loss, val_acc = test(net, data, config)
            net.train(True)
            print("Epoch: ", epoch + 1, "\tTrain Loss: ", epoch_loss/len(train_loader.dataset), "\tVal Loss: ", val_loss)

            wandb.log({ "epoch"      : epoch, 
                        "Train Loss" : epoch_loss/len(train_loader.dataset),
                        "Train Acc"  : correct/total_tested,
                        "Val Loss"   : val_loss,
                        "Val Acc"    : val_acc})

    # Test
    val_loss, val_acc = test(net, data, config)
    wandb.log({"epoch"        : epoch, 
                "Train Loss"  : epoch_loss/len(train_loader.dataset),
                "Train Acc"   : correct/total_tested,
                "Val Loss"    : val_loss,
                "Val Acc"     : val_acc})
    
def test(net, data, config):
    # Set to test mode
    net.train(False)

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


if __name__ == "__main__":
    # Hyperparameters
    gpu          = True 
    save_model   = True
    project_name = "DimahNet CIFAR10"
    set_name     = "CIFAR10"
    # seed         = 100
    # os.environ['WANDB_MODE'] = 'dryrun'

    # Push to GPU if necessary
    if gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    # Declare seed and initalize network
    # torch.manual_seed(seed)

    # Load data
    data = Data(gpu = gpu, set_name = set_name) #, desired_image_size = 224, test_batch_size = 64)
    print(set_name + " is Loaded")

    # Run the sweep
    sweep_id = wandb.sweep(sweep_config, entity="naddeok", project=project_name)
    wandb.agent(sweep_id, function=lambda: train(data, save_model))


