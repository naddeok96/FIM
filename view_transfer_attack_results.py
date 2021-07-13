import xlrd
import matplotlib.pyplot as plt

set_name = "CIFAR10"
attack_type = ["FGSM"]


for attack in attack_type:
    wb = xlrd.open_workbook("results/" + set_name + "/" + attack_type + "_attack_results.xls")
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

    plt.title(attack_type + " attacks on " + set_name)
    plt.xlabel("SNR")
    plt.ylabel("Fooling Ratio")
    plt.legend()
plt.show()