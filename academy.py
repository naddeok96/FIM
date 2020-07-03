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

    def orthogonal_matrix_generator(self, input_tensor):
        '''
        input the model outputs [batches, classes]
        generate orthoganal matrix U the size of classes
        '''
        # Find batch size and feature map size
        num_batches = input_tensor.size(0)
        num_classes = input_tensor.size(1)

        # Calculate an orthoganal matrix the size of A
        U = torch.nn.init.orthogonal_(torch.empty(num_classes, num_classes))
        torch.save(U, "U_from_uni_train")

        # Push to GPU if True
        U = U if self.gpu == False else U.cuda()

        # Repeat U and U transpose for all batches
        Ut = U.t().view((1, num_classes, num_classes)).repeat(num_batches, 1, 1)
        U = U.view((1, num_classes, num_classes)).repeat(num_batches, 1, 1)

        return U
    
    def unitary_cross_entropy(self, input_tensor, target, U):
        '''
        Takes the cross entropy loss of the input oftmaxed and left muiltipled by an orthoganal matrix (U)
        with respect to the target
        '''
        # Find batch size and feature map size
        num_batches = input_tensor.size(0)
        num_classes = input_tensor.size(1)

        # Left muiltiply batches of softmax output  witht the orthogonal matrix
        left_mult_product = torch.bmm(U, F.log_softmax(input_tensor, 1).view((num_batches, num_classes , 1))).view((num_batches, num_classes))
        
        # Return the cross entropy loss wrt the target
        return F.nll_loss(left_mult_product, target)

    def unitary_train(self, batch_size = 124, 
                            n_epochs = 1, 
                            learning_rate = 0.001, 
                            momentum = 0.9, 
                            weight_decay = 0.0001,
                            frozen_layers = None):
        '''
        Train model on train set images with a unitary cross-entropy loss
        '''
        #Get training data
        train_loader = self.data.get_train_loader(batch_size)
        n_batches = len(train_loader)

        #Create optimizer function
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

                # Generate a unitary matrix 
                if i == 0 and epoch == 0:
                    U = self.orthogonal_matrix_generator(outputs)
                    
                # Take the unitary cross entropy loss
                if i == n_batches - 1:
                    loss = self.unitary_cross_entropy(outputs, labels, U[range(outputs.size(0)), :, :]) # Calculate loss
                else:
                    loss = self.unitary_cross_entropy(outputs, labels, U)

                # Freeze layers
                if frozen_layers != None:
                    self.freeze(frozen_layers) 

                # Backward pass and optimize
                loss.backward()                   # Find the gradient for each parameter
                optimizer.step()                  # Parameter update

    def unitary_test(self, U):
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

            # Adjust sizes
            batch_U = U.view(1, 10, 10).repeat(outputs.size(0), 1, 1)
            outputs = outputs.view(outputs.size(0), outputs.size(1), 1)

            # Find unitary output
            uni_output = torch.bmm(batch_U, outputs)
            _, predicted = torch.max(uni_output.data, 1)

            # Update runnin sum
            total_tested += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate accuracy
        accuracy = (correct/total_tested)
        return accuracy

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

        #Create optimizer and loss functions
        optimizer = torch.optim.SGD(self.net.parameters(), 
                                    lr = learning_rate, 
                                    momentum = momentum,
                                    weight_decay = weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

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
                loss = criterion (outputs, labels) # Calculate loss

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

