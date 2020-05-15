'''
This code will be used as the main code to run all classes
'''
# Imports
import torch
from adjustable_lenet import AdjLeNet
from data_setup import Data
from academy import Academy
from ossa import OSSA
import torchvision.transforms.functional as F
import operator
import numpy as np

# Hyperparameters
gpu = False
set_name = "MNIST"
save_set = False
EPSILON = 8

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize data
data = Data(gpu, set_name)

# Load pretraind MNIST network
net = AdjLeNet(set_name = set_name, num_kernels_layer3 = 100)
net.load_state_dict(torch.load('mnist_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
net.eval()

# Generate Attacks
ossa = OSSA(net, data, EPSILON = 0.05, gpu = False)
ossa.get_attacks()






# correct = 0
# adv_correct = 0
# i = 0
# tricked = []
# not_tricked = []
# total_tested = len(data.test_set)
# for inputs, labels in data.test_loader:
#     for image, label in zip(inputs, labels):
#         # Break for iterations
#         if i >= total_tested:
#             break
#         i += 1

#         # Reshape
#         image = image.unsqueeze(0)
#         label = torch.tensor([label.item()])

#         # Initialize InfoGeo object
#         info_geo = InfoGeo(net, image, label, EPSILON = EPSILON)

#         # Calculate FIM
#         info_geo.get_FIM() 

#         # Calculate Attack
#         info_geo.get_attack()

#         # Add to running sum
#         correct += (info_geo.predicted == label).item()
#         adv_correct += (info_geo.adv_predicted == label).item()

#         # Display if tricked
#         if (info_geo.predicted == label).item() == True and (info_geo.adv_predicted == label).item() == False:
#             '''
#             data.plot_attack(image,                  # Image
#                                 info_geo.predicted,     # Prediction
#                                 info_geo.attack,        # Attack
#                                 info_geo.adv_predicted) # Adversrial Prediction
            
#             print("Tricked:", info_geo.FIM_eig_values[0][0].item())
#             '''
#             tricked.append(info_geo.FIM_eig_values[0][0].item())

                
#         elif (info_geo.predicted == label).item() == True and (info_geo.adv_predicted == label).item() == True:
#             #print("Not Tricked:", info_geo.FIM_eig_values[0][0].item())
#             not_tricked.append(info_geo.FIM_eig_values[0][0].item())

# print("================================================================")
# print("Total Tested: ", total_tested)
# print("Model Accuracy: ", correct/total_tested)
# print("Attack Accuracy: ", adv_correct/total_tested)
# print("----------------------------------------------------------------")
# print("Mean of Tricked Eigen Values: ", np.mean(tricked))
# print("Mean of Not Tricked Eigen Values: ", np.mean(not_tricked))
# print("----------------------------------------------------------------")
# print("Std of Tricked Eigen Values: ", np.std(tricked))
# print("Std of Not Tricked Eigen Values: ", np.std(not_tricked))
# print("----------------------------------------------------------------")
# print("Max of Tricked Eigen Values: ", np.max(tricked))
# print("Max of Not Tricked Eigen Values: ", np.max(not_tricked))
# print("----------------------------------------------------------------")
# print("Min of Tricked Eigen Values: ", np.min(tricked))
# print("Min of Not Tricked Eigen Values: ", np.min(not_tricked))
# print("================================================================")



# # Save adverserial image set
# if save_set == True:
#     f = open("adverserial_image_set.txt","w")
#     f.write( str(adverserial_image_set) )
#     f.close()
#     f = open("label_set.txt","w")
#     f.write( str(label_set) )
    # f.close()




