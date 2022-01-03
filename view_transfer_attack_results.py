import xlrd
import matplotlib.pyplot as plt

set_name =  "MNIST" #  "CIFAR10" # 
attack_type = ["FGSM"]
experiment_type = "weakUvsNoU" # "defense comparison" # "UvsNoU"


for attack in attack_type:
    if experiment_type == "defense comparison":
        wb = xlrd.open_workbook("/home/naddeok5/FIM/results/" + set_name + "/" + attack + "/defense_comparison.xls")
    elif experiment_type == "UvsNoU":
        wb = xlrd.open_workbook("/home/naddeok5/FIM/results/" + set_name + "/" + attack + "/UvsNoU_attack_results.xls")
    elif experiment_type == "weakUvsNoU":
        wb = xlrd.open_workbook("/home/naddeok5/FIM/results/" + set_name + "/" + attack + "/weakUvsNoU_attack_results.xls")
    else:
        print("Invalid experiment type.")
        exit()
    # wb = xlrd.open_workbook("results/" + set_name + "/Test_" + attack + "_attack_results.xls")
    ws = wb.sheet_by_name("Results")

    results = {}
    for col in range(len(ws.row(0))):
        results.update({ws.cell(0, col).value: []})

        for row in range(1, len(ws.col(col))):
            results[ws.cell(0, col).value].append(ws.cell(row, col).value)
            
    for key, value in results.items():
        if key == "Epsilons":
            continue

        plt.plot(results["Epsilons"], value, '.-', label=key)

    plt.title(attack + " attacks on " + set_name)
    plt.xlabel("Noise to Signal Ratio")
    plt.ylabel("Fooling Ratio")
    plt.legend()


    # plt.show()
    if experiment_type == "defense comparison":
        plt.savefig("results/" + set_name + "/" + attack + "/defense_comparison_plot.png")

    elif experiment_type == "weakUvsNoU":
        plt.savefig("results/" + set_name + "/" + attack + "/weak_U_fooling_ratio_plot.png")

    elif experiment_type == "UvsNoU":
        plt.savefig("results/" + set_name + "/" + attack + "/fooling_ratio_plot.png")
    else:
        print("Invalid experiment type")
        exit()