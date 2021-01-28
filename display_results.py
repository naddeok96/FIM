
# Imports
import os
import torch
import pickle5 as pickle
from data_setup import Data
from xlrd import open_workbook
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from adversarial_attacks import Attacker
from models.classes.adjustable_lenet import AdjLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet

# Load Data
workbook = open_workbook("10_Unet_transfer_attack_results.xls")
results  = workbook.sheet_by_name("Results")

with open("models/pretrained/10_Rs" + '.pkl', 'rb') as input:
    Rs = pickle.load(input)

# Initialize Data Structure
data = {"Epsilons" : [],
        "Rotations": Rs,
        "OSSA":{"Attacker" : {},
                "LeNet"    : {},
                "U1"       : {},
                "U2"       : {},
                "U3"       : {},
                "U4"       : {},
                "U5"       : {},
                "U6"       : {},
                "U7"       : {},
                "U8"       : {},
                "U9"       : {},
                "U10"      : {}},
        "FGSM":{"Attacker" : {},
                "LeNet"    : {},
                "U1"       : {},
                "U2"       : {},
                "U3"       : {},
                "U4"       : {},
                "U5"       : {},
                "U6"       : {},
                "U7"       : {},
                "U8"       : {},
                "U9"       : {},
                "U10"      : {}}}

# Cycle through the data
for i in range(results.row_len(0)):
    # Epsilons
    if "epsilons" in results.cell_value(0, i):
        data["Epsilons"] = results.col_values(i, start_rowx = 1)

    # Attacker
    if "ossa_fool_ratio" == results.cell_value(0, i):
        data["OSSA"]["Attacker"][results.cell_value(0, i)] = results.col_values(i, start_rowx = 1)

    if "fgsm_fool_ratio" == results.cell_value(0, i):
        data["FGSM"]["Attacker"][results.cell_value(0, i)] = results.col_values(i, start_rowx = 1)

    # Standard
    if "lenet" in results.cell_value(0, i):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["LeNet"][results.cell_value(0, i)] = results.col_values(i, start_rowx = 1)

        if "fgsm" in results.cell_value(0, i):
            data["FGSM"]["LeNet"][results.cell_value(0, i)] = results.col_values(i, start_rowx = 1)

    # Unets
    if str(i - 4) in results.cell_value(0, i):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["U"+str(i - 4)] = results.col_values(i, start_rowx = 1)

        if "fgsm" in results.cell_value(0, i):
            data["FGSM"]["U"+str(i - 4)] = results.col_values(i, start_rowx = 1)

plt.plot(data["Epsilons"], data["OSSA"]["U1"])
plt.show()

    