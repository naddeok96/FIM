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
                       CONVERGE_LIMIT = 0.0001,
                       OCCILLATION_Limit = 0.0001,
                       EPSILON = 0.05,
                       gpu = False):

        super(InfoGeo,self).__init__()

        # Constants for attack
        self.CONVERGE_LIMIT = CONVERGE_LIMIT
        self.OCCILLATION_Limit = OCCILLATION_Limit
        self.EPSILON = EPSILON

        # Move inputs to CPU or GPU
        self.gpu = gpu
        self.net   = net if self.gpu == False else net.cuda()
        self.image = Variable(image, requires_grad = True) if self.gpu == False else Variable(image.cuda(), requires_grad = True)
        self.label = label if self.gpu == False else label.cuda()

        # Evaluation Tools
        self.criterion = torch.nn.CrossEntropyLoss()
        self.soft_max = torch.nn.Softmax(dim = 1)

        # Calculate softmax output
        self.output = self.net(self.image)
        self.soft_max_output = self.soft_max(self.output)

        # Initialize Attack
        self.perturbation = "empty"
        self.attack = "empty"

        # Initialize then calculate FIM
        self.FIM = "empty"
        self.FIM_eig_values = "empty"
        self.FIM_eig_vectors = "empty"

    def unload(self, image):
        '''
        This function unloads an image off a dataloader
        '''
        return image.squeeze(0).squeeze(0)
        
    def get_FIM(self):
        '''
        This function will calculate the FIM
        '''
        # Calculate the loss and gradient of the loss wrt the image for every possible label
        losses = []
        grads_of_losses_wrt_image = []
        for i in range(len(self.output.data[0])):
            # Cycle through lables (y)
            label = torch.tensor([i]) if self.gpu == False else torch.tensor([i]).cuda()

            # Calculate losses
            loss = self.criterion(self.output, label)
            loss.backward(retain_graph = True)
            losses.append(loss.item())
            grads_of_losses_wrt_image.append(self.unload(self.image.grad.data))
        self.loss = losses[self.label.item()]

        # Calculate the FIM
        fisher = 0  
        for i in range(len(self.output.data[0])):
            p = self.soft_max_output.squeeze(0)[i].item()
            g = grads_of_losses_wrt_image[i]
            fisher += p * (torch.t(g) * g)

        self.FIM = fisher
        self.FIM_eig_values, self.FIM_eig_vectors = torch.eig(fisher, eigenvectors = True)

    def get_attack(self):
        '''
        Generate an one step speectral attack
        '''
        # Set the unit norm of the signs of the highest eigenvector to epsilon
        #perturbation = (np.sign(self.FIM_eig_vectors[0]) / np.linalg.norm(np.sign(self.FIM_eig_vectors[0]))) * self.EPSILON
        perturbation = (self.FIM_eig_vectors[0] / np.linalg.norm(self.FIM_eig_vectors[0])) * self.EPSILON
        
        # Check Sign
        attack = self.image + perturbation
        adv_output = self.net(attack)
        adv_loss = self.criterion(adv_output, self.label)

        self.perturbation = perturbation if adv_loss > self.loss else -perturbation
        self.attack = self.image + self.perturbation
        
    def get_prediction(self):
        # See prediction 
        output = self.net(self.image)
        _, self.predicted = torch.max(output.data, 1)

        adv_output = self.net(self.attack)
        _, self.adv_predicted = torch.max(adv_output.data, 1)
        
        
    def plot_attack(self):
        '''
        Plots the image, perturbation and attack
        '''
        # Decalre figure size, figure and title
        figsize = [8, 4]
        fig= plt.figure(figsize=figsize)
        fig.suptitle('OSSA Attack Summary', fontsize=16)

        # Plot orginal image
        ax1 = fig.add_subplot(131)
        ax1.imshow(self.unload(self.image.detach().numpy()), cmap='gray', vmin=0, vmax=255)
        ax1.set_xlabel("Prediction: " + str(self.predicted.item()))
        ax1.set_title("Orginal Image")
        
        # Plot perturbation
        ax2 = fig.add_subplot(232)
        ax2.imshow(self.unload(self.attack.detach().numpy()) - self.unload(self.image.detach().numpy()), cmap='gray', vmin=0, vmax=255)
        ax2.set_title("Attack Perturbation")

        # Plot perturbation unscaled
        ax3 = fig.add_subplot(235)
        ax3.imshow(self.unload(self.attack.detach().numpy()) - self.unload(self.image.detach().numpy()), cmap='gray')
        ax3.set_xlabel("Attack Perturbation Unscaled")
        
        # Plot attack
        ax4 = fig.add_subplot(133)
        ax4.imshow(self.unload(self.attack.detach().numpy()), cmap='gray', vmin=0, vmax=255)
        ax4.set_xlabel("Prediction: " + str(self.adv_predicted.item()))
        ax4.set_title("Attack")

        # Display figure
        plt.show()
        
    

    
        
