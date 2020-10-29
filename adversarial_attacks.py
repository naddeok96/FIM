'''
This class implements adversarial attacks
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
       
        return fisher, losses, predicted

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

    def get_OSSA_attack_accuracy(self, epsilons = [1],
                                       transfer_network = None,
                                       U = None):
        """Determine the accuracy of the network after it is attacked by OSSA

        Returns:
            Attack Accuracy
        """
        # Test images in test loader
        attack_accuracies = np.zeros(len(epsilons))
        for inputs, labels in self.data.test_loader:

            # Push to gpu
            if self.gpu == True:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Calculate FIM
            fisher, losses, predicted = self.get_FIM(inputs, labels)

            # Highest Eigenvalue and vector
            eig_val_max, eig_vec_max = self.get_eigensystem(fisher, max_only = True)

            # Create U(eta) attack
            if U is not None:
                batch_size = eig_vec_max.size(0)
                batch_U = U.view((1, 784, 784)).repeat(batch_size, 1, 1)
                eig_vec_max = torch.bmm(batch_U, eig_vec_max.view(batch_size , 784, 1)).view(batch_size , 1, 784)

            # Cycle over all espiplons
            for i, epsilon in enumerate(epsilons):
                # Set the unit norm of the highest eigenvector to epsilon
                perturbations = epsilon * eig_vec_max

                # Declare attacks as the perturbation added to the image
                batch_size = np.shape(inputs)[0]
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

                if transfer_network == None:
                    adv_outputs = self.net(attacks)
                else:
                    adv_outputs = transfer_network(attacks)

                _, adv_predicted = torch.max(adv_outputs.data, 1)     

                # Save Attack Accuracy
                attack_accuracies[i] = torch.sum(adv_predicted == labels).item() + attack_accuracies[i]
                break
            break
                
        # Divide by total
        attack_accuracies = attack_accuracies / (len(self.data.test_loader.dataset))
                                                        
        return attack_accuracies
  
    def get_single_OSSA_attack(self,    epsilon = 1,
                                        image_index = 0,
                                        plot = False,
                                        eigens_only = False):
        """Get OSSA attack for a single input

        Args:
            epsilon(float, optional): Magnitude of attack. Defaults to 1.
            image_index (int, optional): Image index in data loader. Defualts to first image
            plot (bool, optional): If True, plot attack. Defaults to False.
            eigens_only(bool, optional): If True, only return eigensystem. Defaults to False

        Returns:
            [tuple]: attack, prediction, adverserial prediction
        """
        # Load image
        image, label, index = self.data.get_single_image(index = image_index)

        # Reshape image and make it a variable which requires a gradient
        image = image.view(1,1,28,28)
        image = Variable(image, requires_grad = True) if self.gpu == False else Variable(image.cuda(), requires_grad = True)

        # Reshape label
        label = torch.tensor([label.item()])

        # Get Fisher Information Matrix
        fim, loss, predicted = self.get_FIM(image, label)

        # Get Eigensystem
        eig_val_max, eig_vec_max = self.get_eigensystem(fim, 
                                                        max_only = True)

        if eigens_only == True:
            return eig_val_max, eig_vec_max

        # Set the unit norm of the signs of the highest eigenvector to epsilon
        perturbation = epsilon * (eig_vec_max.view(28*28,1) / torch.norm(eig_vec_max.view(28*28,1)))

        # Calculate sign of perturbation
        attack = (image.view(28*28,1) + perturbation).view(1,1,28,28)

        adv_output = self.net(attack)
        adv_loss = self.criterion(adv_output, label).item()

        perturbation = perturbation if adv_loss > loss else -1*perturbation

        # Compute attack and models prediction of it
        attack = (image.view(28*28,1) + perturbation).view(1,1,28,28)

        adv_output = self.net(attack)
        _, adv_predicted = torch.max(adv_output.data, 1)

        # Display
        if plot == True:
            self.data.plot_attack(image,            # Image
                                    predicted,      # Prediction
                                    attack,         # Attack
                                    adv_predicted)   # Adversrial Prediction

        return attack, predicted, adv_predicted

    def get_gradients(self, images, labels):
        """Calculate the gradients of an image

        Args:
            images: Images to be tested
            labels: Correct lables of images

        Returns:
            gradients, batch_size, num_classes, losses, predicted
        """
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

    def get_FGSM_attack_accuracy(self, epsilons = [1],
                                       transfer_network = None):
        """Generate attacks with FGSM 

        Args:
            EPSILON (int, optional): Magnitude of Attack. Defaults to 1.
            transfer_network (Pytoch Model, optional): Network to have attack transfered to. Defaults to None.

        Returns:
            Float: Attack Accuracy
        """
        # Test images in test loader
        attack_accuracies = np.zeros(len(epsilons))
        for inputs, labels in self.data.test_loader:
            # Push to gpu
            if self.gpu == True:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Calculate FIM
            gradients, batch_size, num_classes, losses, predicted = self.get_gradients(inputs, labels)
            
            # Set the unit norm of the highest eigenvector to epsilon
            gradients_norms = torch.norm(gradients, dim = 1).view(-1, 1, 1).detach()

            for i, epsilon in enumerate(epsilons):
                perturbations = (epsilon * F.normalize(np.sign(gradients), p = 2, dim = 1)).view(batch_size, 1, 28*28)
                
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

                if transfer_network == None:
                    adv_outputs = self.net(attacks)
                else:
                    adv_outputs = transfer_network(attacks)

                _, adv_predicted = torch.max(adv_outputs.data, 1)     

                # Save Attack Accuracy
                attack_accuracies[i] = torch.sum(adv_predicted == labels).item() + attack_accuracies[i]
                break
            break
            
        # Divide by 
        attack_accuracies = attack_accuracies / (len(self.data.test_loader.dataset))

        return attack_accuracies


    def get_fool_ratio(self, test_acc, attack_accs):
        return [(test_acc - attack_acc) / test_acc for attack_acc in attack_accs]
        
    
        
        