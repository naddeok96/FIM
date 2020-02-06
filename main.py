'''
This code will be used as the main code to run all classes
'''

# Imports
from Adjustable_LeNet import AdjLeNet
from MNIST_Setup import MNIST_Data
from Model_Trainer import Model_Trainer
import osos.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# Initialize
net = AdjLeNet(num_classes = 10,
                        num_kernels_layer1 = 6, 
                        num_kernels_layer2 = 16, 
                        num_kernels_layer3 = 120,
                        num_nodes_fc_layer = 84)

data = MNIST_Data()

model_trainer = Model_Trainer(net = net,
                              data = data)

# Fit Model
results1 = model_trainer.train(batch_size = 124, 
                              n_epochs = 1)
results2 = model_trainer.train(batch_size = 124, 
                              n_epochs = 1)
print(results1, results2)

