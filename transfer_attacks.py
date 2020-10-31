# Imports
import os
from adversarial_attacks import Attacker
from models.classes.adjustable_lenet import AdjLeNet
from models.classes.first_layer_unitary_lenet   import FstLayUniLeNet

# Hyperparameters
gpu = False
set_name = "MNIST"
epsilons = [x/5 for x in range(21)]

# Declare which GPU PCI number to use
if gpu == True:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Initialize table
table = PrettyTable()
table.add_column("Epsilons", epsilons)

# Initialize data
data = Data(gpu, set_name)

# Load LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# Unet
Unet = FstLayUniLeNet(set_name = set_name)
Unet.load_state_dict(torch.load('models/pretrained/mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu')))
Unet.U = torch.load('models/pretrained/U_mnist_fstlay_uni_const_lenet_w_acc_95.pt', map_location=torch.device('cpu'))
Unet.eval()

# Load Attacker LeNet
attacker_lenet = AdjLeNet(set_name = set_name)
attacker_lenet.load_state_dict(torch.load('models/pretrained/seed100_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
attacker_lenet.eval()

# Create an attacker
attacker = OSSA(attacker_lenet, data, gpu)

# get_OSSA_attack_accuracy(self, epsilons = [1],
#                                        transfer_network = None,
#                                        U = None):

# Get attack accuracies for Lenet
print("Working on LeNet OSSA Attacks...")
lenet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = lenet_ossa_accs)                                              
lenet_ossa_fool_ratio = attacker.get_fool_ratio(test_acc, lenet_ossa_accs)
table.add_column("LeNet OSSA Fool Ratio", lenet_ossa_fool_ratio)

print("Working on LeNet FGSM Attacks...")
lenet_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = lenet_ossa_accs)
lenet_fgsm_fool_ratio = attacker.get_fool_ratio(test_acc, lenet_fgsm_accs)
table.add_column("LeNet FGSM Attack Accuracy", lenet_fgsm_fool_ratio)


# Get attack accuracies for Unet
print("Working on UNet OSSA Attacks...")
Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet)                                              
Unet_ossa_fool_ratio = attacker.get_fool_ratio(test_acc, Unet_ossa_accs)
table.add_column("UNet OSSA Fool Ratio", Unet_ossa_fool_ratio)

print("Working on UNet FGSM Attacks...")
Unet_fgsm_accs = attacker.get_FGSM_attack_accuracy(epsilons = epsilons,
                                                    transfer_network = Unet)
Unet_fgsm_fool_ratio = attacker.get_fool_ratio(test_acc, Unet_fgsm_accs)
table.add_column("UNet FGSM Attack Accuracy", Unet_fgsm_fool_ratio)


# Display
print(table)

# Excel Workbook Object is created 
if save_to_excel:
    import xlwt 
    from xlwt import Workbook  

    # Open Workbook
    wb = Workbook() 
    
    # Create sheet
    sheet = wb.add_sheet('Results') 

    # Write out each peak and data
    results = [epsilons, ossa_fool_ratio, U_eta_fool_ratio, fgsm_fool_ratio]
    names = ["epsilons", "lenet_ossa_fool_ratio", "lenet_fgsm_fool_ratio", "Unet_ossa_fool_ratio", "Unet_fgsm_fool_ratio"]
    for i, result in enumerate(results):
        sheet.write(0, i, names[i])

        for j, value in enumerate(result):
            sheet.write(j + 1, i, value)

    wb.save('transfer_attack_results.xls') 
