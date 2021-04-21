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
class LitAttacker: 
    def __init__(self, net):

        super(LitAttacker,self).__init__()

        # Evaluation Tools 
        self.criterion = torch.nn.CrossEntropyLoss()
        self.indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        self.soft_max = torch.nn.Softmax(dim = 1)

    def get_max_eigenpair(self, images, labels, max_iter = int(1e4)):
        """Use Lanczos Algorthmn to generate eigenvector associated with the highest eigenvalue

        Args:
            tensor (Tensor): matrix with which eigenvector is desired from
        """
        # Evaluation Tools 
        criterion = torch.nn.CrossEntropyLoss()
        indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
        soft_max = torch.nn.Softmax(dim = 1)

        # Declare Similarity Metric
        mse_sim = torch.nn.MSELoss(reduction='none')
        cos_sim = torch.nn.CosineSimilarity()

        # Make images require gradients
        images.requires_grad_(True)

        #Forward pass
        outputs = self.net.forward(images)
        soft_max_output = self.soft_max(outputs)
        losses = self.indv_criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)

        # Find size parameters
        image_size  = images.size(-1)
        batch_size  = outputs.size(0)
        num_classes = outputs.size(1)

        # Initilize Eigenvector
        eigenvector0 = torch.rand(batch_size, image_size**2, 1).cuda()
        norms = torch.linalg.norm(eigenvector0, ord=2, dim=1).view(-1, 1, 1)
        eigenvector0 = torch.bmm(1 / norms, eigenvector0.view(batch_size, 1, -1)).view(-1, image_size**2, 1)
                
        eigenvector = torch.zeros(batch_size, image_size**2, 1).cuda()

        # Oterate until convergence
        for i in range(max_iter):
            # Calculate expectation
            for i in range(num_classes):
                # Clear Gradients
                self.net.zero_grad()
                images.grad = None

                # Cycle through lables (y)
                temp_labels = torch.tensor([i]).repeat(batch_size) 

                # Calculate losses
                temp_loss = self.criterion(outputs, temp_labels)
                temp_loss.backward(retain_graph = True)

                # Accumulate expectation
                p = soft_max_output[:,i].view(batch_size, 1, 1)
                grad = images.grad.data.view(batch_size, image_size**2, 1)
                
                # p * (gT * eta) * g
                eigenvector += p * (torch.bmm(torch.transpose(grad, 1, 2), eigenvector0) * grad)

            # Normalize
            norms = torch.linalg.norm(eigenvector, ord=2, dim=1).view(-1, 1, 1)
            eigenvector = torch.bmm(1 / norms, eigenvector.view(batch_size, 1, -1)).view(-1, image_size**2, 1)
            
            # Check Convegence
            similarity = torch.mean(cos_sim(eigenvector0.view(-1, 1), 
                                    eigenvector.view(-1, 1))).item()

            if similarity > 0.997:
                # Return vector
                return eigenvector.view(batch_size, 1, -1)

            else:
                # Restart Cycle
                eigenvector0 = eigenvector
                eigenvector = torch.zeros(batch_size, image_size**2, 1).cuda()

        print("Lanczos did not converge...")
        exit()

    def get_OSSA_attack_accuracy(self, inputs, labels,
                                       epsilons = [1],
                                       transfer_network = None,
                                       return_attacks_only = False):
        """Determine the accuracy of the network after it is attacked by OSSA

        Returns:
            Attack Accuracy
        """
        # Test images in test loader
        attack_accuracies = np.zeros(len(epsilons))

        # Calculate FIM then Eigendecomp
        # fisher, losses, predicted = self.get_FIM(inputs, labels)
        # eig_val_max, eig_vec_max = self.get_eigensystem(fisher, max_only = True)
        
        # Highest Eigenvalue's eigenvector from Lanczos Method
        eig_vec_max = self.get_max_eigenpair(inputs, labels)

        # Cycle over all espiplons
        for i, epsilon in enumerate(epsilons):
            # Set the unit norm of the highest eigenvector to epsilon
            perturbations = epsilon * self.normalize(eig_vec_max, p = float('inf'), dim = 2)

            # Declare attacks as the perturbation added to the image
            batch_size = np.shape(inputs)[0]
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            # Check if loss has increased
            adv_outputs = self.net.forward(attacks)
            adv_losses  = self.indv_criterion(adv_outputs, labels)

            # If losses has not increased flip direction
            signs = (losses < adv_losses).type(torch.float) 
            signs[signs == 0] = -1
            perturbations = signs.view(-1, 1, 1) * perturbations
        
            # Compute attack and models prediction of it
            attacks = (inputs.view(batch_size, 1, 28*28) + perturbations).view(batch_size, 1, 28, 28)

            if return_attacks_only:
                return attacks

            if transfer_network == None:
                adv_outputs = self.net.forward(attacks)
            else:
                adv_outputs = transfer_network.forward(attacks)

            _, adv_predicted = torch.max(adv_outputs.data, 1)     

            # Save Attack Accuracy
            attack_accuracies[i] = torch.sum(adv_predicted == labels).item() + attack_accuracies[i]
            
        # Divide by total
        attack_accuracies = attack_accuracies / (batch_size)
                                                        
        return attack_accuracies, batch_size

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
        loss.cpu().backward(retain_graph = True)
        gradients = images.grad.data.view(batch_size, 28*28, 1)      
       
        return gradients, batch_size, num_classes, losses, predicted

    def get_FGSM_attack_accuracy(self, epsilons = [1],
                                       transfer_network = None,
                                       return_attacks_only = False,
                                       attack_images = None,
                                       attack_labels  = None):
        """Generate attacks with FGSM 

        Args:
            EPSILON (int, optional): Magnitude of Attack. Defaults to 1.
            transfer_network (Pytoch Model, optional): Network to have attack transfered to. Defaults to None.

        Returns:
            Float: Attack Accuracy
        """
        # Push transfer_network to GPU
        if self.gpu and (transfer_network is not None):
            transfer_network = transfer_network.cuda()
            
        # Test images in test loader
        attack_accuracies = np.zeros(len(epsilons))
        for inputs, labels in self.data.test_loader:

            if attack_images is not None:
                inputs = attack_images
                labels = attack_labels

            # Push to gpu
            if self.gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Calculate FIM
            gradients, batch_size, num_classes, losses, predicted = self.get_gradients(inputs, labels)

            for i, epsilon in enumerate(epsilons):
                perturbations = (epsilon * self.normalize(torch.sign(gradients), p = np.inf, dim = 1)).view(batch_size, 1, 28*28)

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

                if return_attacks_only:
                    return attacks

                if transfer_network == None:
                    adv_outputs = self.net(attacks)
                else:
                    adv_outputs = transfer_network(attacks)

                _, adv_predicted = torch.max(adv_outputs.data, 1)     

                # Save Attack Accuracy
                attack_accuracies[i] = torch.sum(adv_predicted == labels).item() + attack_accuracies[i]
            
        # Divide by 
        attack_accuracies = attack_accuracies / (len(self.data.test_loader.dataset))

        return attack_accuracies

    def get_fool_ratio(self, test_acc, attack_accs):
        """Calculate the fooling ratio of attacks

        Args:
            test_acc (float): orginal network accuracy
            attack_accs (list of floats): list of accuracies after an attack

        Returns:
            list of floats: list of fooling ratios
        """
        return [round(100*((test_acc - attack_acc) / test_acc), 2) for attack_acc in attack_accs]
        
    def check_attack_perception(self, epsilons = [1]):

        # Initalize images and labels for one of each number
        images = torch.zeros((10, 1, 28, 28))
        labels = torch.zeros((10)).type(torch.LongTensor)
        
        # Find one of each number
        found = False
        number = 0
        index = 0

        while found is not True:
            for test_inputs, test_labels in self.data.test_loader:
                
                image, label, = test_inputs[0], test_labels[0]

                if label.item() == number:
                    images[number, 0, :, :] = image
                    labels[number] = label

                    number += 1
                    if number > 9:
                        found = True
                index += 1


        figsize = 7
        fig, axes2d = plt.subplots(nrows=len(epsilons),
                                    ncols=10,
                                    sharex=True, sharey=True)

        plt.suptitle("MNIST", fontsize=20)
        fig.text(0.02, 0.5, 'L-inf Norm of Attack', va='center', ha='center', rotation='vertical', fontsize=20)

        for i, row in enumerate(axes2d):
            attacks = self. get_OSSA_attack_accuracy(attack_images = images,
                                                            attack_labels = labels,
                                                            epsilons = [epsilons[i]],
                                                            return_attacks_only = True)
            # attacks = self. get_FGSM_attack_accuracy(attack_images = images,
            #                                                 attack_labels = labels,
            #                                                 epsilons = [epsilons[i]],
            #                                                 return_attacks_only = True)

            for j, cell in enumerate(row):
                cell.imshow(attacks[j,:,:,:].detach().numpy()[0], cmap='gray')
                cell.set_xticks([])
                cell.set_yticks([])

                if i == 0:
                    cell.set_title(j)
                if j == 0:
                    cell.set_ylabel(epsilons[i])

        # horz_padding = -0.5
        # vert_padding = -1
        # plt.tight_layout(h_pad=horz_padding, w_pad=vert_padding)
        fig.subplots_adjust(hspace = 0, wspace=0)
            
        
        plt.show()
        exit()
        