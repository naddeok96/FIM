'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
from torch.autograd import Variable
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy
from information_geometry import InfoGeo
import torchvision.transforms.functional as F
import operator
import numpy as np

# Hyperparameters
gpu = False
set_name = "MNIST"
plot = True
save_set = False
EPSILON = 100

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize data
data = Data(gpu, set_name)

# Load pretraind MNIST network
net = AdjLeNet(set_name = set_name)
net.load_state_dict(torch.load('mnist_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
net.eval()

# Evaluation Tools
criterion = torch.nn.CrossEntropyLoss()
soft_max = torch.nn.Softmax(dim = 1)

# Generate Attacks
correct = 0
adv_correct = 0
j = 0
tricked_eig_value = []
not_tricked_eig_value = []
total_tested = 1#len(data.test_set)
for inputs, labels in data.test_loader:
    for image, label in zip(inputs, labels):
        # Break for iterations
        if j >= total_tested:
            break
        j += 1

        # Reshape image and make it a variable which requires a gradient
        image = image.view(1,1,28,28)
        image = Variable(image, requires_grad = True) if gpu == False else Variable(image.cuda(), requires_grad = True)
        
        # Reshape label
        label = torch.tensor([label.item()])

        # Calculate Orginal Loss
        output = net(image)
        soft_max_output = soft_max(output)
        loss = criterion(output, label).item()
        _, predicted = torch.max(soft_max_output.data, 1)

        # Calculate FIM
        fisher = 0 
        for i in range(len(output.data[0])):
            # Cycle through lables (y)
            temp_label = torch.tensor([i]) if gpu == False else torch.tensor([i]).cuda()

            # Reset the gradients
            net.zero_grad()
            image.grad = None

            # Calculate losses
            temp_loss = criterion(output, temp_label)
            temp_loss.backward(retain_graph = True)
            
            # Calculate expectation
            p = soft_max_output.squeeze(0)[i].item()
            fisher += p * (image.grad.data.view(28*28,1) * torch.t(image.grad.data.view(28*28,1)))

        # Calculate Eigenvalues and vectors
        eig_values, eig_vectors = torch.eig(fisher, eigenvectors = True)
        
        # Set the unit norm of the signs of the highest eigenvector to epsilon
        perturbation = EPSILON * (eig_vectors[0].view(28*28,1) / torch.norm(eig_vectors[0].view(28*28,1)))

        # Calculate sign of perturbation
        attack = (image.view(28*28,1) + perturbation).view(1,1,28,28)

        adv_output = net(attack)
        adv_loss = criterion(adv_output, label).item()

        perturbation = perturbation if adv_loss > loss else -perturbation

        # Compute attack and models prediction of it
        attack = (image.view(28*28,1) + perturbation).view(1,1,28,28)

        adv_output = net(attack)
        _, adv_predicted = torch.max(adv_output.data, 1)       

        # Add to running sum
        correct += (predicted == label).item()
        adv_correct += (adv_predicted == label).item()

        if (predicted == label).item() == True and (adv_predicted == label).item() == True:
            not_tricked_eig_value.append(eig_values[0][0])
        elif (predicted == label).item() == True and (adv_predicted == label).item() == False:
            tricked_eig_value.append(eig_values[0][0])

        # Display
        if plot == True:
            data.plot_attack(image,                  # Image
                            predicted,     # Prediction
                            attack,        # Attack
                            adv_predicted) # Adversrial Prediction

# Ensure list are non empty
if len(tricked_eig_value) == 0:
    tricked_eig_value.append(0)
if len(not_tricked_eig_value) == 0:
    not_tricked_eig_value.append(0)

# Display
print("================================================")
print("Total Tested: ",  total_tested)
print("Model Accuracy: ", correct/total_tested)
print("Adverserial Accuracy: ", adv_correct/total_tested)
print("------------------------------------------------")
print("Mean Eigenvalues of Not Tricked: ", np.mean(not_tricked_eig_value))
print("Mean Eigenvalues of Tricked:     ", np.mean(tricked_eig_value))
print("------------------------------------------------")
print("Std Eigenvalues of Not Tricked: ", np.std(not_tricked_eig_value))
print("Std Eigenvalues of Tricked:     ", np.std(tricked_eig_value))
print("------------------------------------------------")
print("Max Eigenvalues of Not Tricked: ", np.max(not_tricked_eig_value))
print("Max Eigenvalues of Tricked:     ", np.max(tricked_eig_value))
print("------------------------------------------------")
print("Min Eigenvalues of Not Tricked: ", np.min(not_tricked_eig_value))
print("Min Eigenvalues of Tricked:     ", np.min(tricked_eig_value))
print("================================================")

        
'''
1. check that other attacks work
2. Calaculte eig by hand then verify that the code works too
3. Check that reshaping is consistent
4. cvx in matlab to solve constrained opt
'''