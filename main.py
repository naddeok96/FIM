'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from mnist_setup import MNIST_Data
from information_geometry import InfoGeo
import torchvision.transforms.functional as F
import operator

# Hyperparameters
gpu = False
plot = False
save_set = False

if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize
data = MNIST_Data()

detministic_model = AdjLeNet(num_classes = 10,
                             num_kernels_layer1 = 6, 
                             num_kernels_layer2 = 16, 
                             num_kernels_layer3 = 120,
                             num_nodes_fc_layer = 84)

detministic_model.load_state_dict(torch.load('trained_lenet_w_acc_98.pt', map_location=torch.device('cpu')))

# Test model
total_tested = 0
correct = 0

# Test images in test loader
for inputs, labels in data.test_loader:

    #Forward pass
    outputs = detministic_model(inputs)
    _, predicted = torch.max(outputs.data, 1)

    total_tested += labels.size(0)
    correct += (predicted == labels).sum().item()
print(correct)
print(total_tested)
print(correct/total_tested)

'''
if save_set == True:
    f = open("adverserial_image_set.txt","w")
    f.write( str(adverserial_image_set) )
    f.close()
    f = open("label_set.txt","w")
    f.write( str(label_set) )
    f.close()
'''



