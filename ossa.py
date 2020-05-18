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
                       EPSILON = 8,
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
        self.indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.soft_max = torch.nn.Softmax(dim = 1)

    def get_attacks(self):
        '''
        Test the model on the unseen data in the test set
        '''
        # Test images in test loader
        attack_data =   {"inputs": "empty",
                        "labels": "empty",
                        "attacks": "empty",
                        "accuracy": "empty",
                        "attack accuracy": "empty"}
        count = 0
        for inputs, labels in self.data.test_loader:
            count =+ 1
            # Push to gpu
            if self.gpu == True:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Make inputs require gradients
            inputs.requires_grad_(True)

            #Forward pass
            outputs = self.net(inputs)
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
                inputs.grad = None

                # Cycle through lables (y)
                temp_labels = torch.tensor([i]).repeat(batch_size) 
                temp_labels = temp_labels if self.gpu == False else temp_labels.cuda()

                # Calculate losses
                temp_loss = self.criterion(outputs, temp_labels)
                temp_loss.backward(retain_graph = True)
                
                # Calculate expectation
                p = soft_max_output[:,i].view(batch_size, 1, 1, 1)
                fisher += p * torch.bmm(inputs.grad.data.view(batch_size, 28*28, 1), torch.transpose(inputs.grad.data.view(batch_size, 28*28, 1), 1, 2)).view(batch_size, 1, 28*28, 28*28)
            
            # Highest Eigenvalue and vector
            eig_values, eig_vectors = torch.symeig(fisher, eigenvectors = True, upper = True)            
            eig_val_max =  eig_values[:, :, -1]
            eig_vec_max = eig_vectors[:, :, :, -1] # already orthonormal

            # Set the unit norm of the signs of the highest eigenvector to epsilon
            perturbations = self.EPSILON * eig_vec_max

            # Declare attacks as the perturbation added to the image
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            # Check if loss has increased
            adv_outputs = self.net(attacks)
            adv_losses  = self.indv_criterion(adv_outputs, labels)

            # If losses has not increased flip direction
            signs = (losses < adv_losses).type(torch.float) 
            signs[signs == 0] = -1
            perturbations = signs.view(batch_size, 1, 1) * perturbations
          
            # Compute attack and models prediction of it
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            adv_outputs = self.net(attacks)
            _, adv_predicted = torch.max(adv_outputs.data, 1)     

            print(count)
            # for name in attack_data:
            #     print(name)
            #     if (name in ["inputs", "labels", "attacks"]) == True and attack_data[name] != "empty":
            #         print(attack_data[name].size())
            #     else:
            #         print(attack_data[name]) 

            # # Save Attack Data
            # if attack_data["inputs"] == "empty": # If First Batch
            #     attack_data["inputs"] = inputs
            #     attack_data["labels"] = labels
            #     attack_data["attacks"] = attacks
            #     attack_data["accuracy"] = torch.sum(predicted == labels).item()
            #     attack_data["attack accuracy"] = torch.sum(adv_predicted == labels).item()
            # else:
            #     attack_data["inputs"] = torch.cat((attack_data["inputs"], inputs), 0)
            #     attack_data["labels"] = torch.cat((attack_data["labels"], labels), 0)
            #     attack_data["attacks"] = torch.cat((attack_data["attacks"], attacks), 0)
            #     attack_data["accuracy"] = torch.sum(predicted == labels).item() + attack_data["accuracy"]
            #     attack_data["attack accuracy"] = torch.sum(adv_predicted == labels).item() + attack_data["attack accuracy"]
            if count >= 2:
                exit()

            
            # return attacks, labels, predicted, adv_predicted 
  
        