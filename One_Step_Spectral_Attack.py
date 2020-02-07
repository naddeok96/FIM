'''
This class implements the One Step Spectral Attack as formulated in the paper
"The Adversarial Attack and Detection under the Fisher Information Metric"
'''

import torch
import operator
from torch.autograd import Variable
import numpy as np

class OSSA:

    def __init__(self, net, 
                       image, 
                       label, 
                       CONVERGE_LIMIT = 0.01):

        self.net =net 
        self.image = image
        self.label = label
        self.CONVERGE_LIMIT = CONVERGE_LIMIT

        self.criterion = torch.nn.CrossEntropyLoss()
        self.soft_max = torch.nn.Softmax(dim = 1)

        self.soft_max_output, self.image_gradients, self.losses = self.get_outputs(self.net, 
                                                                           self.image)

         # Initialize Attack
        self.attack_perturbation = torch.rand(self.image.size()).squeeze(0).squeeze(0)

        self.get_attack_elements()
        
    def get_outputs(self, net,
                          image):

        image = Variable(image, requires_grad = True)
        output = net(image)
        soft_max_output = self.soft_max(output)

        losses = image_gradients = {}
        for i in range(10):
            label = torch.tensor([i])
            loss = self.criterion(output, label)
            loss.backward(retain_graph = True)
            losses[i] = loss.item()

            image_gradients[i] = image.grad.data.squeeze(0).squeeze(0)
            
        return soft_max_output, image_gradients, losses

    def get_attack_elements(self):

        alias_table = [self.soft_max_output]

        converged = False
        dist = 100
        while converged == False:
            expectation = self.get_expectation()
            updated_attack = expectation / torch.norm(expectation)

            updated_dist = torch.dist(self.attack_perturbation, updated_attack).item()

            if updated_dist < self.CONVERGE_LIMIT or dist - updated_dist < 0.01: 
                converged = True

            dist = updated_dist
            self.attack_perturbation = updated_attack
            
    def get_expectation(self):

        expectation = 0  
        for i in range(10):
            p = self.soft_max_output.squeeze(0)[i].item()
            g = self.image_gradients[i]
            expectation += p * (torch.t(g)*self.attack_perturbation) * g


        return expectation
        
    

    
        
