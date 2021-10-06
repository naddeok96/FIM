import xlrd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


set_name = "MNIST"
attack_type = ["Gaussian_Noise", "FGSM"] # , "PGD", "CW2", "OSSA"]
for attack in attack_type:
    for filename in os.listdir("results/" + set_name):
        if attack in filename[:16] and ".xls" in filename:
            print(filename)
            wb = xlrd.open_workbook("results/" + set_name + "/" + filename)
            ws = wb.sheet_by_name("Results")

            results = {}
            for col in range(len(ws.row(0))):
                results.update({ws.cell(0, col).value: []})

                for row in range(1, len(ws.col(col))):
                    if ws.cell(row, 0).value > 0.5:
                        continue
                    results[ws.cell(0, col).value].append(ws.cell(row, col).value)
                    
            for key, value in results.items():
                if key == "NSR":
                    continue

                if "distilled" in filename:
                    label = "Distilled Net"
                    c = "red"
                elif "_U_" in filename:
                    label = "Unitary Net"
                    c = "orange"
                elif "PGD" in filename[5:]:
                    label = "AT-PGD Net"
                    c = "green"
                else:
                    label = "White Box"
                    c = "blue"

                if attack == "FGSM":
                    plt.plot(results["NSR"], value, label=label, color=c)
                else:
                    plt.plot(results["NSR"], value, '.', label=label, color=c)

            plt.title(attack + " on " + set_name)
            plt.xlabel("Noise to Signal Ratio")
            plt.ylabel("Fooling Ratio")
            plt.ylim(top = 100)
            plt.legend()


    # plt.show()
    plt.savefig('results/' + set_name + "/plots/FGSM_GN_results.png")
    # plt.savefig('results/' + set_name + "/plots/" + attack + "_results.png")
    # plt.cla()