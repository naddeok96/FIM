import xlrd
import matplotlib.pyplot as plt


# wb = xlrd.open_workbook("results/MNIST/transfer_attack_results.xls")
wb = xlrd.open_workbook("results/MNIST/nonLanczos_transfer_attack_results.xls")
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
    
plt.legend()
plt.show()