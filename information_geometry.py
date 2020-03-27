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
class InfoGeo:
    def __init__(self, net, 
                       image,
                       label,
                       EPSILON = 0.05,
                       gpu = False):

        super(InfoGeo,self).__init__()

        # Constants for attack
        self.EPSILON = EPSILON

        # Move inputs to CPU or GPU
        self.gpu = gpu
        self.net   = net if self.gpu == False else net.cuda()
        self.image = Variable(image, requires_grad = True) if self.gpu == False else Variable(image.cuda(), requires_grad = True)
        self.label = label if self.gpu == False else label.cuda()

        # Evaluation Tools
        self.criterion = torch.nn.CrossEntropyLoss()
        self.soft_max = torch.nn.Softmax(dim = 1)

        # Calculate Loss
        self.output = self.net(self.image)
        self.soft_max_output = self.soft_max(self.output)
        _, self.predicted = torch.max(self.soft_max_output.data, 1)
        self.loss = self.criterion(self.output, self.label).item()

        # Initialize Attack
        self.perturbation = "empty"
        self.attack = "empty"

        # Initialize then calculate FIM
        self.FIM = "empty"
        self.FIM_eig_vectors = "empty"
        
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
        

        
    

    
        
