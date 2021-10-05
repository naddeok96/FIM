import xlrd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


set_name = "MNIST"
attack_type = ["FGSM", "Gaussian Noise", "CW2", "PGD", "OSSA"]


for attack in attack_type:
    wb = xlrd.open_workbook("results/" + set_name + "/Test_" + attack + "_attack_results.xls")
    ws = wb.sheet_by_name("Results")

    results = {}
    for col in range(len(ws.row(0))):
        results.update({ws.cell(0, col).value: []})

        for row in range(1, len(ws.col(col))):
            # if row > 16:
            #     continue
            results[ws.cell(0, col).value].append(ws.cell(row, col).value)
            
    for key, value in results.items():
        if key == "NSR":
            continue

        plt.plot(results["NSR"], value, '.-', label=attack)

    plt.title("Attacks on " + set_name)
    plt.xlabel("Noise to Signal Ratio")
    plt.ylabel("Fooling Ratio")
    plt.legend()


# plt.show()
plt.savefig('TEMP.png')