# Imports
import os
import io
from unitary_data_setup import UnitaryData
import torch
from models.classes.first_layer_unitary_net import FstLayUniNet
import numpy as np
from data_setup import Data
import re
import operator


# Hypers
root   = '../../../data/naddeok/mnist_U_files/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/train/'

done = np.zeros((int(6e4)),dtype=bool)

for i in os.listdir(root):
    done[int(i.split('U')[-1])] = True
    
s = ''.join('O' if p else 'X' for p in done)
a = list([m.span()[0], abs(operator.sub(*m.span()))] for m in re.finditer('X+', s))
print(a)