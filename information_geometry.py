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
class InformationGeometry:
    def __init__(self, net, 
                       images, 
                       labels, 
                       CONVERGE_LIMIT = 0.0001,
                       OCCILLATION_Limit = 0.0001,
                       EPSILON = 0.05,
                       gpu = False):

        super(InformationGeometry,self).__init__()

        # Move inputs to CPU or GPU
        self.gpu = gpu
        self.net   = net if self.gpu == False net.cuda()
        self.image = Variable(image, requires_grad = True) if self.gpu == False Variable(image.cuda(), requires_grad = True)
        self.label = label if self.gpu == False label.cuda()

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

        # Initialize then calculate FIM
        self.FIM = "empty"
        
    def get_softmaxs_grads_and_losses(self):
        '''
        This function will calculate the softmax output of the model for the image,
        the loss wrt to the image for every label ie grad_J_x(yi,x) for every label i,
        and the loss for each label i
        '''
        # Calculate softmax output
        output = self.net(self.image)
        soft_max_output = self.soft_max(output)

        # Calculate the loss and gradient of the loss wrt the image for every possible label
        losses = {}
        image_gradients = {}
        for i in range(10):
            # Cycle through lables (y)
            label = torch.tensor([i]) if self.gpu == False else torch.tensor([i]).cuda()

            # Calculate losses
            loss = self.criterion(output, label)
            loss.backward(retain_graph = True)
            losses[i] = loss.item()
            grads_of_losses_wrt_image[i] = self.image.grad.data.squeeze(0).squeeze(0)

        fisher = 0  
        for i in range(10):
            p = soft_max_output.squeeze(0)[i].item()
            g = grads_of_losses_wrt_image.cpu()
            fisher += p * (torch.t(g) * g)

        self.FIM = fisher
        
    def get_FIM(self):
        '''
        This function calculates the FIM of the model/input
        '''
        fisher = 0  
        for i in range(10):
            p = self.soft_max_output.squeeze(0)[i].item()
            g = self.image_gradients[i].cpu()
            fisher += p * (torch.t(g)*self.attack_perturbation) * g

        self.FIM = fisher

    def plot_attack(self):
        '''
        Plots the image, perturbation and attack
        '''
        # Declare rows and cols of subplots
        rows = 1
        cols = 3

        # Decalre figure size, figure and title
        figsize = [8, 4]
        fig= plt.figure(figsize=figsize)
        fig.suptitle('OSSA Attack Summary', fontsize=16)

        # Plot orginal image
        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.imshow(show_image, cmap='gray')
        ax1.set_title("Orginal Image")

        # Plot perturbation
        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.imshow(attack.attack_perturbation, cmap='gray')
        ax2.set_title("Attack Perturbation")

        # Plot attack
        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.imshow(show_image + attack.attack_perturbation, cmap='gray')
        ax3.set_title("Attack")

        # Display figure
        plt.show()
        
    

    
        
