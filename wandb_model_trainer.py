'''
This script will train a model and save it
'''
# Imports
import os
import torch
import pickle
from data_setup import Data
# from sweep_config import sweep_config
from lenet_cifar10_sweep_config import sweep_config
from wandb_academy import WandB_Academy
from academy import Academy
from models.classes.rand_lenet                  import RandLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet
from models.classes.first_layer_unitary_vgg     import FstLayUniVGG16
from models.classes.first_layer_rand_lenet      import FstLayRandLeNet
from models.classes.unitary_lenet               import UniLeNet
from models.classes.adjustable_lenet            import AdjLeNet

# Hyperparameters
gpu          = True
project_name = "LeNet CIFAR10"
set_name     = "CIFAR10"
seed         = 100
# os.environ['WANDB_MODE'] = 'dryrun'


# Push to GPU if necessary
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Declare seed and initalize network
torch.manual_seed(seed)

# Load data
data = Data(gpu = gpu, set_name = set_name)

# Enter student network and curriculum data into an academy
academy  = WandB_Academy(project_name,
                         sweep_config,
                         data,
                         gpu)
academy.sweep()
