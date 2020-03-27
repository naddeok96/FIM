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

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad

    # Calculate Norm of perturbation
    per_norm = torch.norm(epsilon * sign_data_grad)

    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad

    # Return the perturbed image
    return perturbed_image, per_norm

# Hyperparameters
gpu = False
set_name = "MNIST"
plot = False
save_set = False
EPSILON = 0.1

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
fooled = 0
j = 0
norms = []
total_tested = 10000#len(data.test_set)
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

        # Reset the gradients
        net.zero_grad()
        image.grad = None

        # Calculate Orginal Loss
        output = net(image)
        soft_max_output = soft_max(output)
        loss = criterion(output, label)
        loss.backward(retain_graph = True)
        _, predicted = torch.max(soft_max_output.data, 1)

        # Generate attack
        attack, per_norm = fgsm_attack(image, EPSILON, image.grad.data)

        # Store norms
        norms.append(per_norm)

        # Test Attack
        adv_output = net(attack)
        _, adv_predicted = torch.max(adv_output.data, 1)       

        # Add to running sum
        correct += (predicted == label).item()
        adv_correct += (adv_predicted == label).item()

        if (predicted == label).item() == True and (adv_predicted == label).item() == False:
            fooled += 1

        # Display
        if plot == True and (predicted == label).item() == True and (adv_predicted == label).item() == False:
            data.plot_attack(image,        # Image
                            predicted,     # Prediction
                            attack,        # Attack
                            adv_predicted) # Adversrial Prediction

# Display
print("================================================")
print("Total Tested: ",  total_tested)
print("Model Accuracy: ", correct/total_tested)
print("Adverserial Accuracy: ", adv_correct/total_tested)
print("Fooling Ratio: ", fooled/correct)
print("Mean of Perturbation Norm: ", np.mean(norms))
print("------------------------------------------------")