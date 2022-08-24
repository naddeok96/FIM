# Imports
from queue import Empty
from data_setup import Data
from unitary_data_setup import UnitaryData
import torch
from models.classes.first_layer_unitary_net import FstLayUniNet
import numpy as np


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
net.load_state_dict(state_dict)
net.eval()

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

# for i, (images, labels) in enumerate(data.test_loader):
# for i, (images, labels) in enumerate(train_loader):
# for i, ((images, labels), (unitary_images, unitary_labels)) in enumerate(zip(train_loader,Utrain_loader)):
correct = 0
custom_unitary_correct = 0
for i, ((images, labels), (unitary_images, unitary_labels)) in enumerate(zip(data.test_loader,Udata.test_loader)):
    if i%1000 == 0:
        print(i)

    if gpu:
        images = images.cuda()
        labels = labels.cuda()
        unitary_images = unitary_images.cuda()
        unitary_labels = unitary_labels.cuda()

    net_used_for_ortho_op.U = torch.load('../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test/' + 'U{}'.format(i))
    custom_unitary_images = net_used_for_ortho_op.orthogonal_operation(images)

    outputs  = net(unitary_images.view(1,1,28,28))
    custom_unitary_outputs  = net(custom_unitary_images.view(1,1,28,28))

    _, predictions  = torch.max(outputs, 1)
    _, custom_unitary_predictions  = torch.max(custom_unitary_outputs, 1)

    correct  = correct    + (labels == predictions).item()
    custom_unitary_correct  = custom_unitary_correct    + (labels == custom_unitary_predictions).item()

print(correct/1e4)
print(custom_unitary_correct/1e4)
    

    
    