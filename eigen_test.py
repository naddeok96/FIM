
# Imports
import torch
from torch.autograd import Variable
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy
from information_geometry import InfoGeo
import torchvision.transforms.functional as F
import operator
import tensorflow as tf
import numpy as np
from scipy import spatial

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


        np.savetxt("FImatrix.CSV",np.asarray(fisher).reshape(1,784*784),delimiter=',')
        exit()

        # Calculate Eigenvalues and vectors
        torch_eig_values, torch_eig_vectors = torch.eig(fisher, eigenvectors = True)

        tf_eig_values, tf_eig_vectors = tf.Session().run(tf.self_adjoint_eig(fisher))

        print("TF eigenvalue:  ",np.max(tf_eig_values))
        print("Torch eigenvalue:  ",torch_eig_values[0][0].item())

        print("\n\n Highest Eigenvector")
        print("Tensorflow       ||       PyTorch")
        for tf, torch in zip(torch.Tensor(tf_eig_vectors[:, np.argmax(tf_eig_values)]).view(28*28,1), torch_eig_vectors[0].view(28*28,1)):
            print(tf.item(),"     ||     " ,torch[0].item())
        
        # print("\n\n Eigenvalues")
        # print("Tensorflow       ||       PyTorch")
        # for tf, torch in zip(np.flip(np.sort(tf_eig_values)),torch_eig_values[:]):
        #     print(tf,"     ||     " ,torch[0].item())