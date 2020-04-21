'''
This class will take a model as a student and a dataset as a curriculum to train/test 
'''
# Imports
import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import math
import operator

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

    def freeze(self, frozen_layers):
        ''' 
        Removes gradients from the layers to be frozen
        '''
        for layer_name ,param in self.net.named_parameters():
                if layer_name in frozen_layers:
                    param.requires_grad = False

    def train(self, batch_size = 124, 
                    n_epochs = 1, 
                    learning_rate = 0.001, 
                    momentum = 0.9, 
                    weight_decay = 0.0001,
                    frozen_layers = None):
        '''
        Train model on train set images
        '''
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

                #Forward pass
                outputs = self.net(inputs)        # Forward pass
                loss = criterion(outputs, labels) # Calculate loss

                # Freeze layers
                if frozen_layers != None:
                    self.freeze(frozen_layers) 

                # Backward pass and optimize
                loss.backward()                   # Find the gradient for each parameter
                optimizer.step()                  # Parameter update
                
    def test(self):
        '''
        Test the model on the unseen data in the test set
        '''
        # Initialize
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

            # Update runnin sum
            total_tested += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        accuracy = (correct/total_tested)
        return accuracy

    def get_single_prediction(self, image):
        '''
        Test model on a single image
        '''
        # Push Image to CPU or GPU 
        image = image if self.gpu == False else image.cuda()

        # Calculate logit outputs
        output = self.net(image)

        # Take max logit to be the predicted class
        _, predicted = torch.max(output.data, 1)
        
        return predicted

