
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

with open("models/pretrained/10_Rs.pkl", 'rb') as input:
    Rs = pickle.load(input)

workbook = open_workbook("High_R_Unet_transfer_attack_results.xls")
bad_results  = workbook.sheet_by_name("Results")

with open("models/pretrained/high_R.pkl", 'rb') as input:
    Bad_R = [pickle.load(input).item()]

# Initialize Data Structure
data = {"Epsilons" : [],
        "Rotations": Bad_R + Rs,
        "OSSA":{"Attacker" : [],
                "LeNet"    : [],
                "Weak Unet": [],
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
                "Weak Unet": [],
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


data["OSSA"]["Weak Unet"] = bad_results.col_values(4, start_rowx = 1)

# Cycle through the data
for i in range(results.row_len(0)):
    # Epsilons
    if "epsilons" in results.cell_value(0, i):
        data["Epsilons"] = results.col_values(i, start_rowx = 1)

    # Attacker
    if "ossa_fool_ratio" == results.cell_value(0, i):
        data["OSSA"]["Attacker"] = results.col_values(i, start_rowx = 1)

    # Standard
    if "lenet" in results.cell_value(0, i):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["LeNet"] = results.col_values(i, start_rowx = 1)

    # Random
    if "random" in results.cell_value(0, i):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["Random"] = results.col_values(i, start_rowx = 1)

    # Unets
    idx = ''.join(filter(lambda i: i.isdigit(), results.cell_value(0, i)))
    if 0 < len(idx):
        if "ossa" in results.cell_value(0, i):
            data["OSSA"]["U"+str(idx)] = results.col_values(i, start_rowx = 1)

plt.style.use('ggplot')

plt.title("Fooling Ratio for Transfer Attacks")
plt.xlabel("Magnitude of Attack Vector")
plt.ylabel("Fooling Ratio")


for model in data["OSSA"]:
    idx = ''.join(filter(lambda i: i.isdigit(), model))
    if model == "Weak Unet":
        plt.plot(data["Epsilons"], data["OSSA"][model], label= "CosSim(I, WeakU): " + "{:.2e}".format(data["Rotations"][0]))

    if 0 < len(idx):
        if idx == "10": 
            plt.plot(data["Epsilons"], data["OSSA"][model], label= "CosSim(I, U" + idx + "): " + "{:.2e}".format(data["Rotations"][int(idx)]))
        else:
            plt.plot(data["Epsilons"], data["OSSA"][model], label= "CosSim(I, U" + idx + ")  : " + "{:.2e}".format(data["Rotations"][int(idx)]))

    else:
        plt.plot(data["Epsilons"], data["OSSA"][model], label= model)

plt.legend(loc = 'lower right', prop={"size":14})
plt.show()
    