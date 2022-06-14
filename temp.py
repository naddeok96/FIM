# Imports
from unitary_data_setup import UnitaryData
import torch
from models.classes.first_layer_unitary_net import FstLayUniNet
import numpy as np

def add_stats(mean1, std1, weight1, mean2, std2, weight2):
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

# Hypers
set_name   = 'MNIST'
batch_size = 1 # int(1e4)
from_ddp   = True
pretrained_weights_filename = "models/pretrained/MNIST/lenet_w_acc_98.pt"
gpu = True
gpu_number = "7"

if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


# Load Source Network
state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

if from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    
net = FstLayUniNet( set_name   = set_name,
                    model_name = "lenet")


data = UnitaryData(test_batch_size=5)

train_loader = data.get_train_loader(batch_size = batch_size, shuffle = False)
tn = 0
for i, (images, labels, unitary_images) in enumerate(train_loader):
    mean_i = unitary_images.mean().detach().numpy()
    std_i  = unitary_images.std().detach().numpy()
    
    if i == 0:
        mean    = mean_i
        std     = std_i
        tn      = 1
    else:
        # Calulate Stats
        mean, std, tn = add_stats(mean_i, std_i, 1,
                                  mean, std, tn)
    # print("Means: ", means)
    # print("Stds: ", stds)
    
    if np.isnan(mean_i):
        print(unitary_images)
        print("image", i)
# for images, labels, unitary_images in data.test_loader:
#     print(images.size(), labels.size(), unitary_images.size())
#     break

print("Mean: ", mean, "STD:", std)
################################################################################################

# import os
# import numpy as np

# not_done = [1] * int(6e4)
# for file in os.listdir("../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/"):
    
#     idx = int(file[1:])
    
#     not_done[idx] = 0
    
# not_done = np.asarray(not_done)

# print(np.sum(not_done))
# print(np.sort(np.argwhere(not_done == 1)))