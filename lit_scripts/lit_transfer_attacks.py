# Imports
from pytorch_lightning.loggers import WandbLogger
from models.classes.lit_lenet import LitLeNet
from lit_adversarial_attacks import LitAttacker
import pytorch_lightning as pl

# Hyperparameters
model_name = "LeNet"
set_name = "MNIST"
wandb_project_name = model_name + " " + set_name + " Lightning"
wandb_mode         = "disabled" # "online", "offline" or "disabled"
run_name           = "mini_standard_U"
gpus               = "4, 5, 6"
save_to_excel = True

epsilons = [x/5 for x in range(61)]

# Initalize Weights and Biases
wandb_logger = WandbLogger(name = run_name, 
                            project=wandb_project_name,
                            mode = wandb_mode)

# Initalize Unitary Model
Unet = LitLeNet.load_from_checkpoint(set_name=set_name,
                                    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_mini_standard_U-val_acc=0.97.ckpt')
Unet.load_orthogonal_matrix("models/pretrained/LeNet_MNIST_mini_standard_U.pkl")


# Initialize Attacker
attacker = LitAttacker(Unet)
ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)               
exit()

# Load LeNet
lenet = AdjLeNet(set_name = set_name)
lenet.load_state_dict(torch.load('models/pretrained/classic_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
lenet.eval()

# UniNet
Unet  = FstLayUniLeNet(set_name = set_name, gpu = gpu)
Unet.load_state_dict(torch.load('models/pretrained/U1_of_10_net_w_acc_95.pt', map_location=torch.device('cpu')))
with open("models/pretrained/" + str(0)+ "_of_" + str(10) + "_Us" + '.pkl', 'rb') as input:
    Unet.U = pickle.load(input).type(torch.FloatTensor)
Unet.eval()

# Weak Uninet
weak_Unet = FstLayUniLeNet(set_name = set_name, gpu = gpu)
weak_Unet.load_state_dict(torch.load('models/pretrained/High_R_Unet_w_acc_98.pt', map_location=torch.device('cpu')))
with open("models/pretrained/high_R_U.pkl", 'rb') as input:
    weak_Unet.U = pickle.load(input).type(torch.FloatTensor)
weak_Unet.eval()

# Load Attacker Net
attacker_lenet = AdjLeNet(set_name = set_name)
attacker_lenet.load_state_dict(torch.load('models/pretrained/seed100_lenet_w_acc_98.pt', map_location=torch.device('cpu')))
attacker_lenet.eval()

# Create an attacker
attacker = Attacker(attacker_lenet, 
                    data, 
                    gpu)

# Get regular attack accuracies on attacker network
print("Working on OSSA Attacks...")
ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons)                                          
ossa_fool_ratio = attacker.get_fool_ratio(0.98, ossa_accs)
table.add_column("OSSA Fool Ratio", ossa_fool_ratio)

# Get transfer attack accuracies for Lenet
print("Working on LeNet OSSA Attacks...")
lenet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = lenet)                                          
lenet_ossa_fool_ratio = attacker.get_fool_ratio(0.98, lenet_ossa_accs)
table.add_column("LeNet OSSA Fool Ratio", lenet_ossa_fool_ratio)

# Unet 
print("Working on UNet OSSA Attacks...")
Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = Unet)                                              
Unet_ossa_fool_ratio = attacker.get_fool_ratio(0.95, Unet_ossa_accs)
table.add_column("UNet OSSA Fool Ratio", Unet_ossa_fool_ratio)

# Weak Unet 
print("Working on Weak UNet OSSA Attacks...")
weak_Unet_ossa_accs  = attacker.get_OSSA_attack_accuracy(epsilons = epsilons,
                                                     transfer_network = weak_Unet)                                              
weak_Unet_ossa_fool_ratio = attacker.get_fool_ratio(0.95, weak_Unet_ossa_accs)
table.add_column("Weak UNet OSSA Fool Ratio", weak_Unet_ossa_fool_ratio)


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
    results = [epsilons, 
               ossa_fool_ratio,
               lenet_ossa_fool_ratio,
               Unet_ossa_fool_ratio,
               weak_Unet_ossa_fool_ratio]
    names = ["epsilons", 
             "ossa_fool_ratio",
             "lenet_ossa_fool_ratio",
             "Unet_ossa_fool_ratio", 
             "weak_Unet_ossa_fool_ratio"]

    for i, result in enumerate(results):
        sheet.write(0, i, names[i])

        for j, value in enumerate(result):
            sheet.write(j + 1, i, value)

    wb.save('High_R_Unet_transfer_attack_results.xls') 
