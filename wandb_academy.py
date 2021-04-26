# Imports
import torch
import math
import copy
import wandb
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet

class WandB_Academy:
    def __init__(self,
                 project_name,
                 sweep_config,
                 data,
                 gpu = False):
        """
        Trains and test models on datasets

        Args:
            net (Pytorch model class):  Network to train and test
            data (Tensor Object):       Data to train and test with
            gpu (bool, optional):       If True then GPU's will be used. Defaults to False.
            autoencoder_trainer (bool, optional): If True labels will equal inputs. Defaults to False.
        """
        super(WandB_Academy,self).__init__()
        # Declare GPU usage
        self.gpu = gpu
        
        # Declare data
        self.data = data

        # Declare Losses
        self.mse_criterion = torch.nn.MSELoss()
        self.cross_entropy_criterion = torch.nn.CrossEntropyLoss()
        self.best_loss = 1e6

        # Declare if Weights and Biases are used
        self.project_name = project_name
        self.sweep_config = sweep_config
        
        
    def sweep(self):
        sweep_id = wandb.sweep(self.sweep_config, entity="naddeok", project=self.project_name)
        wandb.agent(sweep_id, self.train)

    def train(self):
        """Fits model to data

        Args:
            batch_size (int, optional):         Number of samples in each batch. Defaults to 124.
            epochs (int, optional):           Number of cycles through the data. Defaults to 1.
            learning_rate (float, optional):    Learning rate coeffiecent in the optimizer. Defaults to 0.001.
            momentum (float, optional):         Momentum coeffiecent in the optimizer. Defaults to 0.9.
            weight_decay (float, optional):     Weight decay in the optimizer. Defaults to 0.0001.
        """
        
        # Weights and Biases Setup
        config_defaults = {
                    'epochs'       : self.sweep_config['parameters']["epochs"]['values'][0],
                    'batch_size'   : self.sweep_config['parameters']["batch_size"]['values'][0],
                    'momentum'     : self.sweep_config['parameters']["momentum"]['values'][0],
                    'weight_decay' : self.sweep_config['parameters']["weight_decay"]['values'][0],
                    'learning_rate': self.sweep_config['parameters']["learning_rate"]['values'][0],
                    'optimizer'    : self.sweep_config['parameters']["optimizer"]['values'][0],
                    'criterion'    : self.sweep_config['parameters']["criterion"]['values'][0],
                    'num_kernels_layer1' : self.sweep_config['parameters']["num_kernels_layer1"]['values'][0],
                    'num_kernels_layer2' : self.sweep_config['parameters']["num_kernels_layer2"]['values'][0],
                    'num_kernels_layer3' : self.sweep_config['parameters']["num_kernels_layer3"]['values'][0],
                    'num_nodes_fc_layer' : self.sweep_config['parameters']["num_nodes_fc_layer"]['values'][0],
                }
        wandb.init(config = config_defaults)
        self.config = wandb.config

        #Get training data
        train_loader = self.data.get_train_loader(self.config.batch_size)

        # Unet
        net = FstLayUniLeNet(set_name = self.data.set_name, gpu = self.gpu,
                                num_kernels_layer1 = self.config.num_kernels_layer1, 
                                num_kernels_layer2 = self.config.num_kernels_layer2, 
                                num_kernels_layer3 = self.config.num_kernels_layer3,
                                num_nodes_fc_layer = self.config.num_nodes_fc_layer)
        # net = FstLayUniVGG16(set_name = self.data.set_name, gpu = self.gpu, U = None)
        # net.set_random_matrix()
        # net.set_orthogonal_matrix()
        # with open("models/pretrained/high_R_U.pkl", 'rb') as input:
        #     net.U = pickle.load(input).type(torch.FloatTensor)
        net.eval()

        self.net = net if self.gpu == False else net.cuda()
        self.net.train(True)

        # Setup Optimzier
        if self.config.optimizer=='sgd':
            optimizer = torch.optim.SGD(self.net.parameters(),lr=self.config.learning_rate, momentum=self.config.momentum,
            weight_decay=self.config.weight_decay)
        if self.config.optimizer=="adadelta":
            optimizer = torch.optim.Adadelta(self.net.parameters(), lr=self.config.learning_rate, 
                                            weight_decay=self.config.weight_decay, rho=self.config.momentum)
        # Setup Criterion
        if self.config.criterion=="mse":
            criterion = self.mse_criterion
        if self.config.criterion=="cross_entropy":
            criterion = self.cross_entropy_criterion
            
        # Loop for epochs
        correct = 0
        total_tested = 0
        for epoch in range(self.config.epochs):    
            epoch_loss = 0
            for i, data in enumerate(train_loader, 0): 
                # Get labels and inputs from train_loader
                inputs, labels = data

                # One Hot the labels
                orginal_labels = copy.copy(labels).long()
                labels = torch.eq(labels.view(labels.size(0), 1), torch.arange(10).reshape(1, 10).repeat(labels.size(0), 1)).float()
                    
                # Push to gpu
                if self.gpu == True:
                    orginal_labels = orginal_labels.cuda()
                    inputs, labels = inputs.cuda(), labels.cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()   

                #Forward pass
                with torch.set_grad_enabled(True):
                    outputs = self.net(inputs)        # Forward pass
                    if self.config.criterion == "cross_entropy":
                        loss = criterion(outputs, orginal_labels) # Calculate loss
                    else:
                        loss = criterion(outputs, labels) # Calculate loss 
                    
                    # Backward pass and optimize
                    loss.backward()                   # Find the gradient for each parameter
                    optimizer.step()                  # Parameter update
                    
                    _, predictions = torch.max(outputs, 1)
                    correct += (predictions == orginal_labels).sum()
                    total_tested += labels.size(0)
                    epoch_loss += loss.item()

            # Display 
            if epoch % (self.config.epochs/10) == 0:
                val_loss, val_acc = self.test()
                print("Epoch: ", epoch + 1, "\tTrain Loss: ", epoch_loss/len(train_loader.dataset), "\tVal Loss: ", val_loss)

                wandb.log({"epoch"  : epoch, 
                            "Train MSE"   : epoch_loss/len(train_loader.dataset),
                            "Train Acc" : correct/total_tested,
                            "Val MSE" : val_loss,
                            "Val Acc": val_acc})

        # Test
        val_loss, val_acc = self.test()
        wandb.log({"epoch"        : epoch, 
                    "Train MSE"   : epoch_loss/len(train_loader.dataset),
                    "Train Acc"   : correct/total_tested,
                    "Val MSE"     : val_loss,
                    "Val Acc"     : val_acc})

        # Save Model
        # if val_loss < self.best_loss:
        #     wandb.save(self.project_name + "_best_run.h5")
        
    def test(self):
        """Test the model on the unseen data in the test set

        Returns:
            float: Mean loss on test dataset
        """
        #Create loss functions
        if self.config.criterion=="mse":
            criterion = self.mse_criterion
        if self.config.criterion=="cross_entropy":
            criterion = self.cross_entropy_criterion

        # Initialize
        total_loss = 0
        correct = 0
        total_tested = 0
        # Test data in test loader
        for i, data in enumerate(self.data.test_loader, 0):
            # Get labels and inputs from train_loader
            inputs, labels = data

            # One Hot the labels
            orginal_labels = copy.copy(labels).long()
            labels = torch.eq(labels.view(labels.size(0), 1), torch.arange(10).reshape(1, 10).repeat(labels.size(0), 1)).float()

            # Push to gpu
            if self.gpu:
                orginal_labels = orginal_labels.cuda()
                inputs, labels = inputs.cuda(), labels.cuda()

            #Forward pass
            outputs = self.net(inputs)
            if self.config.criterion == "cross_entropy":
                loss = criterion(outputs, orginal_labels) # Calculate loss
            else:
                loss = criterion(outputs, labels) # Calculate loss 

            # Update runnin sum
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == orginal_labels).sum()
            total_tested += labels.size(0)
            total_loss   += loss.item()
        
        # Test Loss
        test_loss = (total_loss/len(self.data.test_set))
        test_acc  = correct / total_tested

        return test_loss, test_acc


