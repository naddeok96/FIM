'''
This script will train a model and save it
'''
# Imports
from adjustable_lenet import AdjLeNet
from mnist_setup import MNIST_Data
from gym import Gym

# Hyperparameters
gpu = True
n_epochs = 10

# Initialize
net = AdjLeNet(num_classes = 10,
               num_kernels_layer1 = 6, 
               num_kernels_layer2 = 16, 
               num_kernels_layer3 = 120,
               num_nodes_fc_layer = 84)
data = MNIST_Data()
detministic_model = Gym(net, data, gpu)

# Fit Model
accuracy = detministic_model.train(n_epochs = n_epochs)

# Save Model
filename = "trained_lenet" + round(accuracy, 2) + ".pt"
torch.save(detministic_model, filename)