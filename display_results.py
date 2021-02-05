
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
        "OSSA":{"Attacker" : [],
                "LeNet"    : [],
                "Random"   : [],
                "U1"       : [],
                "U2"       : [],
                "U3"       : [],
                "U4"       : [],
                "U5"       : [],
                "U6"       : [],
                "U7"       : [],
                "U8"       : [],
                "U9"       : [],
                "U10"      : []},
        "FGSM":{"Attacker" : [],
                "LeNet"    : [],
                "Random"   : [],
                "U1"       : [],
                "U2"       : [],
                "U3"       : [],
                "U4"       : [],
                "U5"       : [],
                "U6"       : [],
                "U7"       : [],
                "U8"       : [],
                "U9"       : [],
                "U10"      : []}}

# Cycle through the data
for i in range(results.row_len(0)):
    # Epsilons
    if "epsilons" in results.cell_value(0, i):
        data["Epsilons"] = results.col_values(i, start_rowx = 1)

    # Attacker
    if "ossa_fool_ratio" == results.cell_value(0, i):
        data["OSSA"]["Attacker"] = results.col_values(i, start_rowx = 1)

    if "fgsm_fool_ratio" == results.cell_value(0, i):
        data["FGSM"]["Attacker"] = results.col_values(i, start_rowx = 1)

    # Standard
    if "lenet" in results.cell_value(0, i):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["LeNet"] = results.col_values(i, start_rowx = 1)

        if "fgsm" in results.cell_value(0, i):
            data["FGSM"]["LeNet"] = results.col_values(i, start_rowx = 1)

    # Random
    if "random" in results.cell_value(0, i):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["Random"] = results.col_values(i, start_rowx = 1)

        if "fgsm" in results.cell_value(0, i):
            data["FGSM"]["Random"] = results.col_values(i, start_rowx = 1)

    # Unets
    idx = ''.join(filter(lambda i: i.isdigit(), results.cell_value(0, i)))
    if 0 < len(idx):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["U"+str(idx)] = results.col_values(i, start_rowx = 1)

        if "fgsm" in results.cell_value(0, i):
            data["FGSM"]["U"+str(idx)] = results.col_values(i, start_rowx = 1)







plt.style.use('ggplot')

plt.title("Fooling Ratio for Transfer Attacks")
plt.xlabel("Magnitude of Attack Vector")
plt.ylabel("Fooling Ratio")


for model in data["OSSA"]:
    idx = ''.join(filter(lambda i: i.isdigit(), model))
        
    if 0 < len(idx):
        if idx == "10": 
            plt.plot(data["Epsilons"], data["OSSA"][model], label= "CosSim(I, U" + idx + "): " + "{:.2e}".format(data["Rotations"][int(idx) - 1]))
        else:
            plt.plot(data["Epsilons"], data["OSSA"][model], label= "CosSim(I, U" + idx + ")  : " + "{:.2e}".format(data["Rotations"][int(idx) - 1]))

    else:
        plt.plot(data["Epsilons"], data["OSSA"][model], label= model)

plt.legend(loc = 'lower right', prop={"size":14})
plt.show()
    