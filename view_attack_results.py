import xlrd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


set_name = "MNIST"
attack_type = ["FGSM"] # ["Gaussian_Noise", "FGSM", "PGD", "CW2", "OSSA"]
upper_bound  = 1

for attack in attack_type:
    for filename in os.listdir("results/" + set_name + "/" + attack + "/"):
        print(filename)
        wb = xlrd.open_workbook("results/" + set_name + "/" + attack + "/" + filename)
        ws = wb.sheet_by_name("Results")

        results = {}
        for col in range(len(ws.row(0))):
            results.update({ws.cell(0, col).value: []})

            for row in range(1, len(ws.col(col))):
                if ws.cell(row, 0).value > upper_bound:
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

            elif "lenet_w_acc_97_on_lenet_w_acc_98_attack_results" in filename:
                label = "LeNet-5"
                c = "red"

            elif "lenet_w_acc_97_on_lenet_w_acc_97_attack_results":
                label = "White Box"
                c = "blue"
            else:
                label = "Unknown"
                c = "black"

            plt.plot(results["NSR"], value, label=label, color=c)
                

    plt.title(attack + " on " + set_name)
    plt.xlabel("Ratio of Noise to Signal L2 Norms")
    plt.ylabel("Fooling Ratio")
    # plt.ylim(top = 100)
    plt.legend()


    # plt.show()
    # plt.savefig('results/' + set_name + "/plots/TEST_FGSM_GN_results.png")
    plt.savefig('results/' + set_name + "/plots/" + attack + "_results.png")
    # plt.cla()