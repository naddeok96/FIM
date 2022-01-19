import xlrd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


set_name      = "CIFAR10"
save_filename = "LSR_U_defense_comparison_side_by_side.png"
attack_type   = ["FGSM", "OSSA"] # ["Gaussian_Noise", "FGSM", "PGD", "CW2", "OSSA"]
files_to_skip = ["UvsNoU", "distill", "weak"]
upper_bound   = 1
num_markers   = 10

fig, ax = plt.subplots(1, len(attack_type))
for i, attack in enumerate(attack_type):
    print(attack)
    ax[i].set_title(attack)

    for filename in os.listdir("results/" + set_name + "/" + attack + "/"):
        # Filter out unwanted files
        if "attack_results" not in filename or any(file_to_skip in filename for file_to_skip in files_to_skip):
            continue
        print(filename[:-19])
        
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
            if key == "NSR" or key == "Epsilons":
                continue

            if "distilled" in filename:
                label = "Distilled Net"
                c = "red"

            elif "_U_" in filename:
                label = "Unitary Net"
                c = "orange"
                m = "o"
                me = int(len(results["NSR"])/num_markers)

            elif "LSR" in filename:
                label = "LSR"
                c = "red"
                m = "s"
                me = int(len(results["NSR"])/num_markers) - 1

            elif "PGD" in filename[5:]:
                label = "AT-PGD Net"
                c = "red"

            elif "lenet_w_acc_97_on_lenet_w_acc_98_attack_results" in filename:
                label = "Black Box"
                c = "blue"
                m = "v"
                me = int(len(results["NSR"])/num_markers) - 2

            elif "lenet_w_acc_97_on_lenet_w_acc_97_attack_results" in filename:
                label = "White Box"
                c = "green"
                m = ","
                me = int(len(results["NSR"])/num_markers) - 3
                
            elif "cifar10_mobilenetv2_x1_0_w_acc_93_on_Nonecifar10_mobilenetv2_x1_0_w_acc_91_attack_results" in filename:
                label = "Black Box"
                c = "blue"
                m = "v"
                me = int(len(results["NSR"])/num_markers) - 2
                
            elif "cifar10_mobilenetv2_x1_0_w_acc_93_on_cifar10_mobilenetv2_x1_0_w_acc_93_attack_results" in filename:
                label = "White Box"
                c = "green"
                m = ","
                me = int(len(results["NSR"])/num_markers) - 3

            else:
                label = "Unknown"
                c = "black"
                
            if label in ["Black Box"]:
                continue

            if "NSR" in results.keys():
                # plt.plot(results["NSR"], value, label=label, color=c)
                ax[i].plot(results["NSR"], value, label=label, color=c, marker=m, markevery=me)
                
            else:
                # plt.plot(results["Epsilons"], value, label=label, color=c)
                ax[i].plot(results["Epsilons"], value, label=label, color=c, marker=m, markevery=me)

fig.suptitle(set_name)

plt.legend()

fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("Ratio of Noise to Signal L2 Norms")
plt.ylabel("Fooling Ratio")

# plt.show()
# plt.savefig('results/' + set_name + "/plots/TEST_FGSM_GN_results.png")
plt.savefig('results/' + set_name + "/" + attack + "/" + save_filename)
plt.cla()