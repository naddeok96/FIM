'''
This class will take a model and a dataset and fit the former to the latter
'''

import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

class Model_Trainer:

    def __init__(self,
                 net, 
                 data):

        self.net = net
        self.data = data

    def train(self, batch_size, 
                    n_epochs, 
                    learning_rate = 0.001, 
                    momentum = 0.9, 
                    weight_decay = 0.0001):

        #Get training data
        train_loader = self.data.get_train_loader(batch_size)
        n_batches = len(train_loader)

        #Create our loss and optimizer functions
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.net.parameters(), 
                                    lr = learning_rate, 
                                    momentum = momentum,
                                    weight_decay = weight_decay)

        #Loop for n_epochs
        for epoch in range(n_epochs):            
            for i, data in enumerate(train_loader, 0):
                
                #Reset the train loader and apply a counter
                inputs, labels = data

                # Push input to gpus
                inputs, labels = inputs.cuda(), labels.cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()
                
                #Forward pass, backward pass, optimize
                outputs = self.net(inputs) # Forward pass
                loss = criterion(outputs, labels) # calculate loss
                loss.backward() # Find the gradient for each parameter
                optimizer.step() # Parameter update
                

        # At the end of training run a test
        total_tested = 0
        correct = 0
        for inputs, labels in self.data.test_loader:
            
            # Push input to gpu
            inputs, labels = inputs.cuda(), labels.cuda()

            #Forward pass
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_tested += labels.size(0)
            correct += (predicted == labels).sum().item()
        return (correct/total_tested)

