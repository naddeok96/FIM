# Imports 
from operator import index
import torch
import os
import time
import matplotlib.pyplot as plt
from data_setup import Data
from models.classes.first_layer_unitary_net import FstLayUniNet

# Hypers
# set_name   = 'MNIST'
# start_image_index = int(0 * 666)
# gpu        = True
# gpu_number = "6"
# batch_size = 1
# from_ddp   = True
# pretrained_weights_filename = "models/pretrained/MNIST/MNIST_Models_for_Optimal_U_stellar-rain-5.pt"

# model_name = pretrained_weights_filename.split('.')[0].split('/')[-1]
# save_path  = "../../../data/naddeok/mnist_U_files/optimal_U_for_" + model_name 
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
#     os.mkdir(save_path + "/train")
#     os.mkdir(save_path + "/test")

# if gpu:
#     import os
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# # Load Data
# data = Data(set_name = set_name,
#             test_batch_size = 1,
#             gpu = gpu)

# # Load Source Network
# state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

# if from_ddp:  # Remove prefixes if from DDP
#     torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    
# net = FstLayUniNet( set_name   = set_name,
#                     model_name = "lenet",
#                     gpu = gpu)

# train_loader = data.get_train_loader(batch_size=1, shuffle=False)
# print("Train Set")
# count_down = 0
# for i, (image, label) in enumerate(train_loader):
#     print("Checking: ", i)
#     # Skip
#     if i < start_image_index: 
#         print("Skip due to start index...")
#         continue
    
#     # Give space
#     if count_down > 0:
#         count_down -= 1
#         print("Skip due to count down...")
#         continue
    
#     # Check if done
#     already_done = False
#     for file in os.listdir(save_path + "/train"):
    
#         if 'U{}'.format(i) == file:
#             already_done = True
            
#             # count_down = 500
#             break

#     if already_done:
#         print("Skip due to already done...")
#         continue
    
#     # # Display if made
#     print("Working on: ", i)
    
#     if gpu:
#         image = image.cuda()
#         label = label.cuda()
        
#     # Set a optimal orthogonal matrix
#     print("Generating orthogonal matrix")
#     net.set_orthogonal_matrix(image)

#     # Save Optimal U for Image
#     torch.save(net.U, save_path + '/train/U{}'.format(i))
        
    # if torch.sum(torch.isnan(torch.load(save_path + '/train/U{}'.format(i).format(i)))) != 0 or torch.sum(torch.isnan(torch.load('../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/UA{}.pt'.format(i)))) != 0:
    #     torch.save(image, 'TEMP/U{}'.format(i))
        
        
        # Set a optimal orthogonal matrix
        # print("Generating orthogonal matrix")
        # net.set_orthogonal_matrix(image)

        # # Save Optimal U for Image
        # torch.save(net.U, '../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/U{}'.format(i))
        
        # Load Optimal U for image
        # net.U = torch.load('../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/U{}'.format(i))
        
        # Rotate Images
        # ortho_image = net.orthogonal_operation(image)
        
        # # # Save
        # torch.save(ortho_image, '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/UA{}.pt'.format(i))
        
        # if torch.isnan(torch.load('../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/U{}'.format(i))).any() or torch.isnan(torch.load('../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/UA{}.pt'.format(i))).any():
        #     print("DID NOT FIX")
            # break

    
# print("Test Set")
# for i, (image, label) in enumerate(data.test_loader):
#     print("Checking: ", i)
#     # Skip
#     if i < start_image_index: 
#         print("Skip due to start index...")
#         continue
    
#     # Give space
#     if count_down > 0:
#         count_down -= 1
#         print("Skip due to count down...")
#         continue
    
#     # Check if done
#     already_done = False
#     for file in os.listdir(save_path + "/test"):
    
#         if 'U{}'.format(i) == file:
#             already_done = True
            
#             count_down = 333
#             break

#     if already_done:
#         print("Skip due to already done...")
#         continue
    
#     # # Display if made
#     print("Working on: ", i)
    
    
    
#     if gpu:
#         image = image.cuda()
#         label = label.cuda()
        
#     # Set a optimal orthogonal matrix
#     print("Generating orthogonal matrix")
#     net.set_orthogonal_matrix(image)

#     # Save Optimal U for Image
#     torch.save(net.U, save_path + '/test/U{}'.format(i))

    # Load Optimal U for image
    # net.U = torch.load('../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/test/U{}'.format(i))
    
    # # Rotate Images
    # ortho_image = net.orthogonal_operation(image)
    
    # # Save
    # torch.save(ortho_image, '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/UA{}.pt'.format(i))

    
# print("donezo")

# Join all images into one dataset
unitary_images = torch.empty((  int(1e4),
                                1,
                                28, 
                                28))

for i in range(original_images.size(0)):
    UA = torch.load(unitary_root + 'UA{}.pt'.format(i))
    unitary_images[i,:,:,:] = UA

torch.save((unitary_images, labels), '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/testing.pt')