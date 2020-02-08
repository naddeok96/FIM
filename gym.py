'''
This class will take a model and a dataset and fit the former to the latter
'''

import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

class Gym:

    def __init__(self,
                 net, 
                 data,
                 gpu = False):

        self.gpu = gpu
        self.net = net if self.gpu == False else net.cuda()
        self.data = data if self.gpu == False else data.cuda()

    def train(self, batch_size = 124, 
                    n_epochs = 1, 
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

            #Forward pass
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_tested += labels.size(0)
            correct += (predicted == labels).sum().item()
        return (correct/total_tested)

    def get_single_prediction(self, image):
        
        # Push Image to GPU 
        image = image if self.gpu == False else image.cuda()

        output = self.net(image)
        _, predicted = torch.max(output.data, 1)
        
        return predicted

