from data_setup import Data
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.classes.first_layer_unitary_net    import FstLayUniNet
import torchvision
import torchvision.transforms as transforms
import os
import art
from art import config
from art.estimators.classification import PyTorchClassifier

# Hypers
set_name = 'CIFAR10'
batch_size = int(5e4)
gpu = True
batch_size = 256

# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Load Data
config.set_data_path('../../../data/pytorch/CIFAR10')
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
# data = Data(gpu = gpu, set_name = set_name, maxmin = True, test_batch_size = batch_size)
# images, labels = next(iter(data.test_loader))
# images, labels = images.detach().numpy(), labels.detach().numpy()
print(set_name + " is Loaded")

# Load Network
net = FstLayUniNet(set_name = set_name, gpu =gpu,
                       model_name = 'cifar10_mobilenetv2_x1_0',
                       pretrained = True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=net,
    clip_values=(data.test_pixel_min, data.test_pixel_max),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(data.num_channels, data.image_size, data.image_size),
    nb_classes=data.num_classes,
)


predictions = classifier.predict(images)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=0)) / len(labels)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))


# Load adverasy
# art.attacks.evasion.CarliniL2Method(classifier = net, 
#                                     confidence =  0.0,
#                                     learning_rate = 0.01, 
#                                     binary_search_steps = 10, 
#                                     max_iter = 10,
#                                     initial_const = 0.01, 
#                                     max_halving = 5,
#                                     max_doubling  = 5, 
#                                     batch_size = 1,
#                                     verbose= True)

# Collect Stats
# for images, labels in dataloader:
    
