'''
This class implements the One Step Spectral Attack as formulated in the paper
"The Adversarial Attack and Detection under the Fisher Information Metric"
'''
# Imports
import torch
import torch.nn.functional as F
import operator
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import copy

# Class
class Attacker: 
    def __init__(self, net, 
                       data,
                       gpu = False):
        """This class will use data to generate attacks on network

        Args:
            net (pytorch model): Neural Network to be attacked
            data (pytorch data loader): Data in a dataloader    
            gpu (bool, optional): Wheather or not to use GPU. Defaults to False.
        """

        super(Attacker,self).__init__()

        # Move inputs to CPU or GPU
        self.gpu = gpu
        self.net   = net if self.gpu == False else net.cuda()
        self.data = data

        # Evaluation Tools 
        self.criterion = torch.nn.CrossEntropyLoss()
        self.indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.soft_max = torch.nn.Softmax(dim = 1)

    def get_FIM(self, images, labels):
        """Calculate the Fisher Information Matrix for all images

        Args:
            images : Images to be used
            labels : Correct labels of images

        Returns:
            FIM, Loss for each Image, Predicted Class for each image
        """
        
        # Push to gpu
        images = Variable(images, requires_grad = True) if self.gpu == False else Variable(images.cuda(), requires_grad = True)
        labels = labels if self.gpu == False else labels.cuda()

        # Make images require gradients
        images.requires_grad_(True)

        #Forward pass
        outputs = self.net(images)
        soft_max_output = self.soft_max(outputs)
        losses = self.indv_criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        # Find size parameters
        batch_size  = outputs.size(0)
        num_classes = outputs.size(1)

        # Calculate FIMs
        fisher = 0 
        for i in range(num_classes):
            # Clear Gradients
            self.net.zero_grad()
            images.grad = None

            # Cycle through lables (y)
            temp_labels = torch.tensor([i]).repeat(batch_size) 
            temp_labels = temp_labels if self.gpu == False else temp_labels.cuda()

            # Calculate losses
            temp_loss = self.criterion(outputs, temp_labels)
            temp_loss.backward(retain_graph = True)

            # Calculate expectation
            p = soft_max_output[:,i].view(batch_size, 1, 1, 1)
            grad = images.grad.data.view(batch_size, 28*28, 1)

            fisher += p * torch.bmm(grad, torch.transpose(grad, 1, 2)).view(batch_size, 1, 28*28, 28*28)
       
        return fisher, num_classes, losses, predicted

    def get_eigensystem(self, tensor, max_only = False):
        """Given a tensor find the eigensystem

        Args:
            tensor (Tensor): A tensor object
            max_only (bool, optional): Wheater to just return the maximum or all. Defaults to False.

        Returns:
            Eigenvalues, Eigenvectors
        """
        # Find eigen system
        eig_values, eig_vectors = torch.symeig(tensor, eigenvectors = True, upper = True)       

        if max_only == True:     
            eig_val_max =  eig_values[:, :, -1]
            eig_vec_max = eig_vectors[:, :, :, -1] 
            return eig_val_max, eig_vec_max

        else:
            return eig_values, eig_vectors

    def get_OSSA_attack_accuracy(self, EPSILON = 1):
        """Determine the accuracy of the network after it is attacked by OSSA

        Returns:
            Attack Accuracy
        """
        # Test images in test loader
        attack_accuracy = 0
        count = 0
        for inputs, labels in self.data.test_loader:
            # Calculate FIM
            fisher, losses, predicted = self.get_FIM(inputs, labels)

            # Highest Eigenvalue and vector
            eig_val_max, eig_vec_max = self.get_eigens(fisher, max_only = True)

            # Set the unit norm of the highest eigenvector to epsilon
            perturbations = EPSILON * eig_vec_max

            # Declare attacks as the perturbation added to the image
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            # Check if loss has increased
            adv_outputs = self.net(attacks)
            adv_losses  = self.indv_criterion(adv_outputs, labels)

            # If losses has not increased flip direction
            signs = (losses < adv_losses).type(torch.float) 
            signs[signs == 0] = -1
            perturbations = signs.view(-1, 1, 1) * perturbations
          
            # Compute attack and models prediction of it
            batch_size = len(inputs.dataset)
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            adv_outputs = self.net(attacks)
            _, adv_predicted = torch.max(adv_outputs.data, 1)     

            # Save Attack Accuracy
            attack_accuracy = torch.sum(adv_predicted == labels).item() + attack_accuracy
            
        # Divide by 
        attack_accuracy = attack_accuracy / (len(self.data.test_loader.dataset))
                                                        
        return attack_accuracy
  
    def get_attack(self, image, label, plot = False):
        """Get OSSA attack for a single input

        Args:
            image (Tensor): Single input image
            label (Tensor): Label for image
            plot (bool, optional): If True, plot attack. Defaults to False.

        Returns:
            [tuple]: attack, prediction, adverserial prediction
        """
        # Reshape image and make it a variable which requires a gradient
        image = image.view(1,1,28,28)
        image = Variable(image, requires_grad = True) if self.gpu == False else Variable(image.cuda(), requires_grad = True)

        # Reshape label
        label = torch.tensor([label.item()])

        # Calculate Orginal Loss
        output = self.net(image)
        soft_max_output = self.soft_max(output)
        loss = self.criterion(output, label).item()
        _, predicted = torch.max(soft_max_output.data, 1)

        # Calculate FIM
        fisher = 0 
        for i in range(len(output.data[0])):
            # Cycle through lables (y)
            temp_label = torch.tensor([i]) if self.gpu == False else torch.tensor([i]).cuda()

            # Reset the gradients
            self.net.zero_grad()
            image.grad = None

            # Calculate losses
            temp_loss = self.criterion(output, temp_label)
            temp_loss.backward(retain_graph = True)

            # Calculate expectation
            p = soft_max_output.squeeze(0)[i].item()
            fisher += p * (image.grad.data.view(28*28,1) * torch.t(image.grad.data.view(28*28,1)))

        # Highest Eigenvalue and vector
        eig_values, eig_vectors = torch.eig(fisher, eigenvectors = True)
        eig_val_max = eig_values[0][0]
        eig_vec_max = eig_vectors[:, 0]

        # Set the unit norm of the signs of the highest eigenvector to epsilon
        perturbation = self.EPSILON * (eig_vec_max.view(28*28,1) / torch.norm(eig_vec_max.view(28*28,1)))

        # Calculate sign of perturbation
        attack = (image.view(28*28,1) + perturbation).view(1,1,28,28)

        adv_output = self.net(attack)
        adv_loss = self.criterion(adv_output, label).item()

        perturbation = perturbation if adv_loss > loss else -perturbation

        # Compute attack and models prediction of it
        attack = (image.view(28*28,1) + perturbation).view(1,1,28,28)

        adv_output = self.net(attack)
        _, adv_predicted = torch.max(adv_output.data, 1)

        # Display
        if plot == True:
            self.data.plot_attack(image,            # Image
                                    predicted,      # Prediction
                                    attack,         # Attack
                                    adv_predicted,  # Adversrial Prediction
                                    self.model_name)# Model Name
        return attack, predicted, adv_predicted

    def get_gradients(self, images, labels):
        '''
        Finds the gradients of model given images and labels wrt to the images
        '''
        # Push to gpu
        images = Variable(images, requires_grad = True) if self.gpu == False else Variable(images.cuda(), requires_grad = True)
        labels = labels if self.gpu == False else labels.cuda()

        # Make images require gradients
        images.requires_grad_(True)

        # Clear Gradients
        self.net.zero_grad()
        images.grad = None

        #Forward pass
        outputs = self.net(images)
        soft_max_output = self.soft_max(outputs)
        loss = self.criterion(outputs, labels)
        losses = self.indv_criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        # Find size parameters
        batch_size  = outputs.size(0)
        num_classes = outputs.size(1)

        # Find gradients
        loss.backward(retain_graph = True)
        gradients = images.grad.data.view(batch_size, 28*28, 1)
       
        return gradients, batch_size, num_classes, losses, predicted

    def get_FGSM_attack_accuracy(self):
        '''
        Test the model on the unseen data in the test set
        '''
        # Test images in test loader
        attack_accuracy = 0
        for inputs, labels in self.data.test_loader:
            # Calculate FIM
            gradients, batch_size, num_classes, losses, predicted = self.get_gradients(inputs, labels)
            
            # Set the unit norm of the highest eigenvector to epsilon
            gradients_norms = torch.norm(gradients, dim = 1).view(-1, 1, 1).detach()

            perturbations = (self.EPSILON * F.normalize(np.sign(gradients), p = 2, dim = 1)).view(batch_size, 1, 28*28)
            
            # Declare attacks as the perturbation added to the image
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            # Check if loss has increased
            adv_outputs = self.net(attacks)
            adv_losses  = self.indv_criterion(adv_outputs, labels)

            # If losses has not increased flip direction
            signs = (losses < adv_losses).type(torch.float) 
            signs[signs == 0] = -1
            perturbations = signs.view(-1, 1, 1) * perturbations
          
            # Compute attack and models prediction of it
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            adv_outputs = self.net(attacks)
            _, adv_predicted = torch.max(adv_outputs.data, 1)     

            # Save Attack Accuracy
            attack_accuracy = torch.sum(adv_predicted == labels).item() + attack_accuracy
            
        # Divide by 
        attack_accuracy = attack_accuracy / (len(self.data.test_loader.dataset))

        return attack_accuracy 
        
        