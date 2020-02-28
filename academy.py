'''
This class will take a model and a dataset and fit the former to the latter
'''

import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

class Academy:
    def __init__(self,
                 net, 
                 data,
                 gpu = False):

        super(Academy,self).__init__()

        # Declare GPU usage
        self.gpu = gpu

        # Push net to CPU or GPU
        self.net = net if self.gpu == False else net.cuda()
        
        # Declare data
        self.data = data

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
                
                # Get inputs and labels from train_loader
                inputs, labels = data

                # Push to gpu
                if self.gpu == True:
                    inputs, labels = inputs.cuda(), labels.cuda()

                #Set the parameter gradients to zero
                optimizer.zero_grad()
                
                #Forward pass, backward pass, optimize
                outputs = self.net(inputs)        # Forward pass
                loss = criterion(outputs, labels) # Calculate loss
                loss.backward()                   # Find the gradient for each parameter
                optimizer.step()                  # Parameter update
                
    def test(self):
        # At the end of training run a test
        total_tested = 0
        correct = 0

        # Test images in test loader
        for inputs, labels in self.data.test_loader:
            # Push to gpu
            if self.gpu == True:
                inputs, labels = inputs.cuda(), labels.cuda()

            #Forward pass
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_tested += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = (correct/total_tested)
        return accuracy

    def get_single_prediction(self, image):
        
        # Push Image to CPU or GPU 
        image = image if self.gpu == False else image.cuda()

        output = self.net(image)
        _, predicted = torch.max(output.data, 1)
        
        return predicted

