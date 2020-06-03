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

# Class
class OSSA: 
    def __init__(self, net, 
                       data,
                       EPSILON = 8,
                       gpu = False,
                       model_name=""):

        super(OSSA,self).__init__()

        # Constants for attack
        self.EPSILON = EPSILON

        # Move inputs to CPU or GPU
        self.gpu = gpu
        self.net   = net if self.gpu == False else net.cuda()
        self.data = data
        self.model_name = model_name

        # Evaluation Tools 
        self.criterion = torch.nn.CrossEntropyLoss()
        self.indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.soft_max = torch.nn.Softmax(dim = 1)


    def add_stats(self, mean1, std1, weight1, mean2, std2, weight2):
        '''
        Takes stats of two sets (assumed to be from the same distribution) and combines them
        Method from https://www.statstodo.com/CombineMeansSDs_Pgm.php
        '''
        # Calculate E[x] and E[x^2] of each
        sig_x1 = weight1 * mean1
        sig_x2 = weight2 * mean2

        sig_xx1 = ((std1 ** 2) * (weight1 - 1)) + (((sig_x1 ** 2) / weight1))
        sig_xx2 = ((std2 ** 2) * (weight2 - 1)) + (((sig_x2 ** 2) / weight2))

        # Calculate sums
        tn  = weight1 + weight2
        tx  = sig_x1  + sig_x2
        txx = sig_xx1 + sig_xx2

        # Calculate combined stats
        mean = tx / tn
        std = np.sqrt((txx - (tx**2)/tn) / (tn - 1))

        return mean, std, tn

    def get_fim(self, images, labels):
        '''
        Finds the Fisher Information Matrix of model given images and labels
        '''
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
            fisher += p * torch.bmm(images.grad.data.view(batch_size, 28*28, 1), torch.transpose(images.grad.data.view(batch_size, 28*28, 1), 1, 2)).view(batch_size, 1, 28*28, 28*28)
       
        return fisher, batch_size, num_classes, losses, predicted

    def get_eigens(self, fisher):
        '''
        Returns the highest eigenvalue and associated eigenvector
        '''
        # Highest Eigenvalue and vector
        eig_values, eig_vectors = torch.symeig(fisher, eigenvectors = True, upper = True)            
        eig_val_max =  eig_values[:, :, -1]
        eig_vec_max = eig_vectors[:, :, :, -1] # already orthonormal

        return eig_val_max, eig_vec_max


    def get_attack_accuracy(self):
        '''
        Test the model on the unseen data in the test set
        '''
        # Test images in test loader
        attack_accuracy = 0
        fooled_max_eig_data = []
        unfooled_max_eig_data = []
        count = 0
        for inputs, labels in self.data.test_loader:
            count += 1
            print(count)
            if count >= 2:
                break
            # Calculate FIM
            fisher, batch_size, num_classes, losses, predicted = self.get_fim(inputs, labels)

            # Highest Eigenvalue and vector
            eig_val_max, eig_vec_max = self.get_eigens(fisher)

            # Set the unit norm of the highest eigenvector to epsilon
            perturbations = self.EPSILON * eig_vec_max

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

            # Determine fooled and unfooled attacks
            results = torch.cat((adv_predicted.type(torch.FloatTensor).view(-1, 1), 
                                 predicted.type(torch.FloatTensor).view(-1, 1), 
                                 labels.type(torch.FloatTensor).view(-1, 1)), 1)
            correct  = results[results[:, 1] == results[:, 2], :] # Where the classifier predicted correctly
            fooled_indices   = [i for i, x in enumerate(correct[:, 0] != correct[:, 2]) if x] # Where the classifier was orginally correct then fooled
            unfooled_indices = [i for i, x in enumerate(correct[:, 0] == correct[:, 2]) if x] # Where the classifer was orginally correct and still correct
            
            # Calaulate attacks FIM
            fooled_fisher,_ ,_ ,_ ,_    = self.get_fim(attacks[fooled_indices], labels[fooled_indices])
            unfooled_fisher,_ ,_ ,_ ,_  = self.get_fim(attacks[unfooled_indices], labels[unfooled_indices])

            # Highest Eigenvalue and vector of attacks
            fooled_eig_val_max, fooled_eig_vec_max     = self.get_eigens(fooled_fisher)
            fooled_eig_vec_max = signs[fooled_indices].view(-1, 1, 1) * fooled_eig_vec_max

            unfooled_eig_val_max, unfooled_eig_vec_max = self.get_eigens(unfooled_fisher)
            unfooled_eig_vec_max = signs[unfooled_indices].view(-1, 1, 1) * unfooled_eig_vec_max

            # Calculate cosine similarity of images and attacks
            fooled_cos_sim = abs(torch.cosine_similarity(fooled_eig_vec_max, perturbations[fooled_indices] , dim=2, eps=1e-6))
            unfooled_cos_sim = abs(torch.cosine_similarity(unfooled_eig_vec_max, perturbations[unfooled_indices] , dim=2, eps=1e-6))

            fooled_max_eig_data.append((fooled_eig_val_max.view(-1).mean().item(), # Mean eigenvalue of image
                                        fooled_eig_val_max.view(-1).std().item(),  # STD eigenvalue of image
                                        fooled_cos_sim.view(-1).mean().item(),     # Mean cosine similarity between image and attack eigenvectors
                                        fooled_cos_sim.view(-1).std().item(),      # STD cosine similarity between image and attack eigenvectors
                                        len(fooled_indices)))
            unfooled_max_eig_data.append((unfooled_eig_val_max.view(-1).mean().item(), 
                                        unfooled_eig_val_max.view(-1).std().item(), 
                                        unfooled_cos_sim.view(-1).mean().item(),    
                                        unfooled_cos_sim.view(-1).std().item(),
                                        len(unfooled_indices)))

            # Save Attack Accuracy
            attack_accuracy = torch.sum(adv_predicted == labels).item() + attack_accuracy
            
        # Divide by 
        attack_accuracy = attack_accuracy / (len(self.data.test_loader.dataset))

        # Calculate avg_fooled_max_eig
        for i, data in enumerate(fooled_max_eig_data):
            if i == 0:
                fooled_max_eig_stats = data

            else:
                val_mean, val_std, num_data = self.add_stats(fooled_max_eig_stats[0],
                                                            fooled_max_eig_stats[1],
                                                            fooled_max_eig_stats[4],
                                                            data[0], data[1], data[4])
                vec_mean, vec_std, num_data = self.add_stats(fooled_max_eig_stats[2],
                                                            fooled_max_eig_stats[3],
                                                            fooled_max_eig_stats[4],
                                                            data[2], data[3], data[4])
                fooled_max_eig_stats = (val_mean, val_std,
                                        vec_mean, vec_std, num_data)

        # Calculate avg_fooled_max_eig
        for i, data in enumerate(unfooled_max_eig_data):
            if i == 0:
                unfooled_max_eig_stats = data

            else:
                val_mean, val_std, num_data = self.add_stats(unfooled_max_eig_stats[0],
                                                                unfooled_max_eig_stats[1],
                                                                unfooled_max_eig_stats[4],
                                                                data[0], data[1], data[4])
                vec_mean, vec_std, num_data = self.add_stats(unfooled_max_eig_stats[2],
                                                                unfooled_max_eig_stats[3],
                                                                unfooled_max_eig_stats[4],
                                                                data[2], data[3], data[4])
                unfooled_max_eig_stats = (val_mean, val_std,
                                        vec_mean, vec_std, num_data)
                                                        
        return attack_accuracy, fooled_max_eig_stats, unfooled_max_eig_stats
  
    def get_attack(self, image, label, plot = False):
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
        