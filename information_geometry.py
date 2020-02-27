'''
This class implements the One Step Spectral Attack as formulated in the paper
"The Adversarial Attack and Detection under the Fisher Information Metric"
'''
import torch
import operator
from torch.autograd import Variable
import numpy as np

class InformationGeometry:
    def __init__(self, net, 
                       images, 
                       labels, 
                       CONVERGE_LIMIT = 0.0001,
                       OCCILLATION_Limit = 0.0001,
                       EPSILON = 0.05,
                       gpu = False):

        self.gpu = gpu
        # Move inputs to CPU or GPU
        if self.gpu == False:
            self.net   = net
            self.image = Variable(image, requires_grad = True)
            self.label = label
        else:
            self.net   = net.cuda()
            self.image = Variable(image.cuda(), requires_grad = True)
            self.label = label.cuda()

        # Constants for attack
        self.CONVERGE_LIMIT = CONVERGE_LIMIT
        self.OCCILLATION_Limit = OCCILLATION_Limit
        self.EPSILON = EPSILON

        # Evaluation Tools
        self.criterion = torch.nn.CrossEntropyLoss()
        self.soft_max = torch.nn.Softmax(dim = 1)

        # Get the output probabilities, loss gradient wrt to input and losses 
        self.soft_max_output, self.image_gradients, self.losses = self.get_outputs()

        # Initialize Attack
        self.attack_perturbation = torch.rand(self.image.size()).squeeze(0).squeeze(0)

        # Calculate models softmax outputs, loss wrt to image and losses
        self.soft_max_output = 0
        self.losses = 0
        self.loss_wrt_image = 0
        self.get_softmaxs_grads_and_losses()

        # Calculate FIM
        self.FIM = 0
        self.get_FIM()
        
    def get_softmaxs_grads_and_losses(self):
        output = self.net(self.image)
        self.soft_max_output = self.soft_max(output)

        losses = image_gradients = {}
        for i in range(10):
            # Cycle through lables (y)
            label = torch.tensor([i]) if self.gpu == False else torch.tensor([i]).cuda()

            # Calculate losses
            loss = self.criterion(output, label)
            loss.backward(retain_graph = True)
            losses[i] = loss.item()
            loss_wrt_image[i] = self.image.grad.data.squeeze(0).squeeze(0)

        self.losses = losses   
        self.loss_wrt_image = loss_wrt_image
        

    def get_FIM(self):
        fisher = 0  
        for i in range(10):
            p = self.soft_max_output.squeeze(0)[i].item()
            g = self.image_gradients[i].cpu()
            fisher += p * (torch.t(g)*self.attack_perturbation) * g

        self.FIM = fisher
        
    

    
        
