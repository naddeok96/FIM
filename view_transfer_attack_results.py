import xlrd

wb = xlrd.open_workbook("results/MNIST/transfer_attack_results.xls")
ws = wb.sheet_by_name("Results")

