'''
This class implements the One Step Spectral Attack as formulated in the paper
"The Adversarial Attack and Detection under the Fisher Information Metric"
'''
# Imports
import torch
import operator
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# Class
class OSSA:
    def __init__(self, net, 
                       data,
                       EPSILON = 0.05,
                       gpu = False):

        super(OSSA,self).__init__()

        # Constants for attack
        self.EPSILON = EPSILON

        # Move inputs to CPU or GPU
        self.gpu = gpu
        self.net   = net if self.gpu == False else net.cuda()
        self.data = data

        # Evaluation Tools
        self.criterion = torch.nn.CrossEntropyLoss()
        self.soft_max = torch.nn.Softmax(dim = 1)

    def get_attacks(self):
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

            # Make inputs require gradients
            inputs.requires_grad_(True)

            #Forward pass
            outputs = self.net(inputs)
            soft_max_output = self.soft_max(outputs)

            # Find size parameters
            batch_size  = outputs.size(0)
            num_classes = outputs.size(1)

            # Calculate FIMs
            grads_of_losses_wrt_image = {}
            fisher = 0 
            for i in range(num_classes):
                # Cycle through lables (y)
                temp_labels = torch.tensor([i]).repeat(batch_size) 
                temp_labels = temp_labels if self.gpu == False else temp_labels.cuda()
                
                # Calculate losses
                inputs.grad = None
                loss = self.criterion(outputs, temp_labels)
                loss.backward(retain_graph = True)
                print(inputs.grad)
                exit()
            
  

        
    def get_FIM(self):
        '''
        This function will calculate the FIM
        '''
        # Calculate FIM
        grads_of_losses_wrt_image = {}
        fisher = 0 
        for i in range(len(self.output.data[0])):
            # Cycle through lables (y)
            label = torch.tensor([i]) if self.gpu == False else torch.tensor([i]).cuda()

            # Calculate losses
            self.image.grad = None
            loss = self.criterion(self.output, label)
            loss.backward(retain_graph = True)
            
            grads_of_losses_wrt_image[i] = self.image.grad.data.view(28*28,1)

            # Calculate expectation
            p = self.soft_max_output.squeeze(0)[i].item()
            tp += p
            fisher += p * (grads_of_losses_wrt_image[i] * torch.t(grads_of_losses_wrt_image[i]))
        
        self.FIM = fisher

        self.FIM_eig_values, self.FIM_eig_vectors = torch.eig(fisher, eigenvectors = True)

    def get_attack(self):
        '''
        Generate a one step spectral attack
        '''
        # Set the unit norm of the signs of the highest eigenvector to epsilon
        perturbation = self.EPSILON * (self.FIM_eig_vectors[0] / torch.norm(self.FIM_eig_vectors[0]))

        # Calculate sign of perturbation
        attack = (self.image.view(28*28) + perturbation).view(1,1,28,28)

        adv_output = self.net(attack)
        adv_loss = self.criterion(adv_output, self.label).item()

        self.perturbation = perturbation if adv_loss > self.loss else -perturbation

        # Compute attack and models prediction of it
        self.attack = (self.image.view(28*28) + self.perturbation).view(1,1,28,28)

        adv_output = self.net(self.attack)
        _, self.adv_predicted = torch.max(adv_output.data, 1)       
        