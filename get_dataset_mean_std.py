# Imports
from queue import Empty
from data_setup import Data
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

    # sig_xx1 = ((std1 ** 2) * (weight1 - 1)) + (((sig_x1 ** 2) / weight1))
    # sig_xx2 = ((std2 ** 2) * (weight2 - 1)) + (((sig_x2 ** 2) / weight2))

    # Calculate sums
    tn  = weight1 + weight2
    tx  = sig_x1  + sig_x2
    # txx = sig_xx1 + sig_xx2

    # Calculate combined stats
    mean = tx / tn
    # std = np.sqrt((txx - (tx**2)/tn) / (tn - 1))
    std = np.sqrt(((weight1*(std1*std1) + weight2*(std2*std2)) + (weight1*((mean1 - mean)*(mean1 - mean))) + (weight2*((mean2 - mean)*(mean2 - mean)))) / (weight1 + weight2))


    return mean, std, tn

# Hypers
set_name   = 'MNIST'
batch_size = 1
from_ddp   = True
pretrained_weights_filename = "models/pretrained/MNIST/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5_MNIST_Models_for_Optimal_U_spring-smoke-13.pt"
gpu = True
gpu_number = "3"

if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Load Source Network
state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

if from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    
net = FstLayUniNet( set_name   = set_name,
                    model_name = "lenet",
                    gpu = gpu)

net_used_for_ortho_op = FstLayUniNet( set_name   = set_name,
                    model_name = "lenet",
                    gpu = gpu)

data = Data(set_name = set_name,
            test_batch_size = batch_size,
            gpu = gpu)

Udata = UnitaryData(set_name = set_name,
                    test_batch_size = batch_size,
                    unitary_root = '../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/',
                    gpu = gpu)

train_loader    = data.get_train_loader(batch_size = batch_size, shuffle = False)
Utrain_loader   = Udata.get_train_loader(batch_size = batch_size, shuffle = False)
tn = 0
unitary_tn = 0
empty_set = torch.empty((int(1e4),1,28,28))
# for i, (images, labels) in enumerate(data.test_loader):
# for i, (images, labels) in enumerate(train_loader):
# for i, ((images, labels), (unitary_images, unitary_labels)) in enumerate(zip(train_loader,Utrain_loader)):
for i, ((images, labels), (unitary_images, unitary_labels)) in enumerate(zip(data.test_loader,Udata.test_loader)):
    if i%1000 == 0:
        print(i)

    if gpu:
        images = images.cuda()
        labels = labels.cuda()
        unitary_images = unitary_images.cuda()
        unitary_labels = unitary_labels.cuda()

    mean_i = images.mean().detach().cpu().numpy()
    std_i  = images.std().detach().cpu().numpy()

    net_used_for_ortho_op.U = torch.load('../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test/' + 'U{}'.format(i))
    custom_unitary_images = net_used_for_ortho_op.orthogonal_operation(images)
    custom_unitary_mean_i = custom_unitary_images.mean().detach().cpu().numpy()
    custom_unitary_std_i  = custom_unitary_images.std().detach().cpu().numpy()

    unitary_mean_i = unitary_images.mean().detach().cpu().numpy()
    unitary_std_i  = unitary_images.std().detach().cpu().numpy()
    
    if i == 0:
        mean    = mean_i
        std     = std_i
        tn      = images.size(0)

        custom_unitary_mean = custom_unitary_mean_i
        custom_unitary_std  = custom_unitary_std_i
        custom_unitary_tn   =  custom_unitary_images.size(0)

        unitary_mean    = unitary_mean_i
        unitary_std     = unitary_std_i
        unitary_tn      =  unitary_images.size(0)
    else:
        # Calculate Stats
        mean, std, tn = add_stats(  mean_i, std_i, images.size(0),
                                    mean, std, tn)

        custom_unitary_mean, custom_unitary_std, custom_unitary_tn = add_stats(custom_unitary_mean_i, custom_unitary_std_i, custom_unitary_images.size(0),
                                                    custom_unitary_mean, custom_unitary_std, custom_unitary_tn)
                                    
        unitary_mean, unitary_std, unitary_tn = add_stats(unitary_mean_i, unitary_std_i, unitary_images.size(0),
                                                    unitary_mean, unitary_std, unitary_tn)
    

print("Mean: ", mean, "STD:", std, "Weight:", tn )
print("Uni Custom Mean: ", custom_unitary_mean, "Uni Custom STD:", custom_unitary_std)
print("Uni Mean: ", unitary_mean, "Uni STD:", unitary_std)
