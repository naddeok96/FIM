# Imports
import os
import torch
from data_setup import Data
from adversarial_attacks import Attacker
from models.classes.first_layer_unitary_net  import FstLayUniNet

# Hyperparameters
set_name      = "MNIST"
model_name    = "lenet" 
batch_size    = 1

attack_net_filename = "models/pretrained/MNIST/lenet_w_acc_97.pt"
attack_net_from_ddp = True

reg_net_filename    = "models/pretrained/MNIST/lenet_w_acc_98.pt" # "models/pretrained/MNIST/distilled_20_lenet_w_acc_94.pt" # "models/pretrained/MNIST/lenet_w_acc_98.pt"
reg_net_from_ddp    = True

# Initialize data
data = Data(set_name = set_name, maxmin = True, test_batch_size = batch_size, root='../data')

# Load Networks
#-------------------------------------------------------------------------------------------------------------------------------------___#
# # Load Attacker Net
attacker_net = FstLayUniNet(set_name,
                            U_filename = None,
                            model_name = model_name)
attack_net_state_dict = torch.load(attack_net_filename, map_location=torch.device('cpu'))
torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(attack_net_state_dict, "module.")
attacker_net.load_state_dict(attack_net_state_dict)
attacker_net.eval()

attacker = Attacker(attacker_net, data)


# Attack Networks
#-----------------------------------------------------------------------------------------------------------------------#
# Create an attacker
images, labels = next(iter(data.get_train_loader(batch_size=1)))

# Let's say we have matrix A and we found its largest eigenvalue lambda_max and corresponding eigenvector. Consider matrix 
# B=lambda_max id - A. Find its largest eigenvalue mu_max and corresponding eigenvector. Then this eigenvector is the smallest eigenvector of A and 
# lambda_min = lambda_max - mu_max.


A, _, _ = attacker.get_FIM(images, labels)
A = A[0,0,:,:]

eigenvalues, eigenvectors = torch.linalg.eig(A)
min_idx = torch.argmin(torch.abs(torch.real(eigenvalues)))
max_idx = torch.argmax(torch.abs(torch.real(eigenvalues)))
A_eigvec_min = torch.real(eigenvectors[:, min_idx])

lambda_min = torch.min(torch.abs(torch.real(eigenvalues)))
print("Lambda Min:", lambda_min)

lambda_max = torch.max(torch.abs(torch.real(eigenvalues)))
B = lambda_max * torch.eye(A.size(-1)) - A

eigenvalues, eigenvectors = torch.linalg.eig(B)
min_idx = torch.argmin(torch.abs(torch.real(eigenvalues)))
max_idx = torch.argmax(torch.abs(torch.real(eigenvalues)))
B_eigvec_max = torch.real(eigenvectors[:, max_idx])

mu_max = torch.max(torch.abs(torch.real(eigenvalues)))

print("lambda_max - mu_max", lambda_max - mu_max)


cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)

print(cossim(A_eigvec_min, B_eigvec_max))
# eigenvalues, eigenvectors = torch.linalg.eig(A)
# min_idx = torch.argmin(torch.abs(torch.real(eigenvalues)))
# max_idx = torch.argmax(torch.abs(torch.real(eigenvalues)))

# eig_min = eigenvectors[:, min_idx]
# eig_max = eigenvectors[:, max_idx]