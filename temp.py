# Imports
import os
import io
from unitary_data_setup import UnitaryData
import torch
from models.classes.first_layer_unitary_net import FstLayUniNet
import numpy as np
from data_setup import Data
import re
import operator




# Imports
import xlwt
from xlwt import Workbook
import torch
import copy

# Try a unitary transformation that swaps first and last eigenvector
# -> First write a matrix in a basis of the highest and lowest eigenvector than convert to real numbers
# -->> Normalize v_max and v_min, then take space spaned by them, check email from Nidhal


# GramSchmidt Algorithm
def GramSchmidt(A):
    # Get eigensystem
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    min_idx = torch.argmin(torch.abs(torch.real(eigenvalues)))
    max_idx = torch.argmax(torch.abs(torch.real(eigenvalues)))
   
    eig_min = eigenvectors[:, min_idx]
    eig_max = eigenvectors[:, max_idx]
    # print("Min/Max", torch.dot(eig_max,eig_max))
    # exit()
    # eig_min = eigenvectors[:, -1]
    # eig_max = eigenvectors[:, 0]

    
    
    # Generate initial matrix to transform into unitary
    V = torch.randn_like(A)
    V[0,:] = torch.real(eig_min)
    V[1,:] = torch.real(eig_max)

    # Orthogonal complement of V in n-dimension 
    for i in range(A.size(0)):
        # Orthonormalize
        Ui = copy.copy(V[i])

        for j in range(i):
            Uj = copy.copy(V[j])

            
            Ui = Ui - ((torch.dot(Uj.view(-1), Ui.view(-1)) / (torch.linalg.norm(Uj, ord = 2)**2)))*Uj
            
        V[i] = Ui / torch.linalg.norm(Ui, ord = 2)

    return V.t()

# Matrix Multiplication
def mm(A, B):
    return torch.mm(A,B)

# Transpose
def t(A):
    return torch.transpose(A, 0, 1)

# Build excel sheet with eigensystem
def build_excel(dict_of_matrices):
    # from openpyxl import load_workbook
    # workbook = load_workbook("EigensystemComparison.csv")
    # sheet = workbook['Eigensystem Comparison']

    # exit()
    # from xlrd import open_workbook
    # from xlutils.copy import copy
    
    # rb = open_workbook("EigensystemComparison.xls")
    # workbook = copy(rb)
    # sheet = workbook.get_sheet("Eigensystem Comparison")

    workbook = xlwt.Workbook() 

    sheet = workbook.add_sheet("Eigensystem Comparison")
    
    for i, key in enumerate(dict_of_matrices):
        # Get matrix 
        M = dict_of_matrices[key]
        nrows = M.size(0)
        ncols = M.size(1)
        
        # Plot Key
        sheet.write(i*(nrows+4) + 2, 0, key)
        
        # Plot matrix
        for j in range(nrows):
            for k in range(ncols):
                sheet.write(i*(nrows+4)+3+j, k, round(M[j,k].item(),2))
                
        # Get eigensystem
        eigenvalues, eigenvectors = torch.linalg.eig(M)
        
        
        # Plot Eigenvalues
        sheet.write(i*(nrows+4) + 2, ncols + 1, "Eigenvalues")
        for j in range(nrows):
            sheet.write(i*(nrows+4) + 3, ncols + 1 + j, round(torch.real(eigenvalues[j]).item(),2))
            
        # Plot Eigenvectors
        borders = xlwt.Borders()
        borders.top = xlwt.Borders.THIN
        borders.top_colour = 0xBFBFBF
        borders.bottom = xlwt.Borders.THIN
        borders.bottom_colour = 0xBFBFBF
        style = xlwt.XFStyle()
        style.borders = borders
        
        
        sheet.write(i*(nrows+4) + 4, ncols + 1, "Eigenvectors")
        for j in range(nrows):
            for k in range(ncols):
                sheet.write(i*(nrows+4)+5+j, ncols + 1 + k, round(torch.real(eigenvectors[k, j]).item(),2), style)
                
    # Save        
    workbook.save("EigensystemComparison.xls")

    import pyexcel as p

    p.save_book_as(file_name="EigensystemComparison.xls",
               dest_file_name="EigensystemComparison.xlsx")
    
# Calculate FIM from Image and Model
def get_FIM(net, images, labels):

    soft_max = torch.nn.Softmax(dim = 1)
    indv_criterion = torch.nn.CrossEntropyLoss(reduction = 'none')
    criterion = torch.nn.CrossEntropyLoss()
    
    # Make images require gradients
    images.requires_grad_(True)

    #Forward pass
    outputs = net(images)
    soft_max_output = soft_max(outputs)
    losses = indv_criterion(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)

    # Find size parameters
    batch_size  = outputs.size(0)
    num_classes = outputs.size(1)

    # Calculate FIMs
    fisher = 0 
    for i in range(num_classes):
        # Clear Gradients
        net.zero_grad()
        images.grad = None

        # Cycle through lables (y)
        temp_labels = torch.tensor([i]).repeat(batch_size) 
        

        # Calculate losses
        temp_loss = criterion(outputs, temp_labels)
        temp_loss.backward(retain_graph = True)

        # Calculate expectation
        p = soft_max_output[:,i].view(batch_size, 1, 1, 1)
        grad = images.grad.data.view(batch_size,  images.size(2)* images.size(2), 1)
        
        fisher += p * torch.bmm(grad, torch.transpose(grad, 1, 2)).view(batch_size, 1,  images.size(2)* images.size(2),  images.size(2)* images.size(2))

    return fisher, losses, predicted

def get_eigensystem(tensor, max_only = False):
    # Find eigen system
    tensor = tensor.cpu()
    eig_values, eig_vectors = torch.symeig(tensor, eigenvectors = True, upper = True)   


    if max_only == True:     
        eig_val_max =  eig_values[:, :, -1]
        eig_vec_max = eig_vectors[:, :, :, -1] 
        return eig_val_max, eig_vec_max

    else:
        return eig_values, eig_vectors


# Main
if __name__ == "__main__":
    # # Generate Random Positive Definite
    # mat_size = 5
    # A = torch.rand(mat_size, mat_size)
    # A = mm(A, t(A))
    # A.add_(torch.eye(mat_size))

    # # Orthogonal matrix with first 2 columns as the eigenvectors associated with the min and max eigenvalues
    # D = GramSchmidt(A)
  
    # # Permutation matrix
    # P = torch.eye(mat_size)
    # index = torch.tensor(range(P.size(0)))
    # index[0:2] = torch.tensor([1, 0])
    # P = P[index]

    # # Basis change of D from P
    # N = mm(mm(D,P),t(D))
    
    # # New matrix with 
    # B = mm(mm(N,A),t(N))

    # Hypers
    pretrained_weights_filename = "models/pretrained/MNIST/MNIST_Models_for_Optimal_U_stellar-rain-5.pt"
    from_ddp = True
    unitary_root    = "../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test/"

    # Load Network
    state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

    if from_ddp:  # Remove prefixes if from DDP
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
        
    net = FstLayUniNet( set_name   = "MNIST",
                        model_name = "lenet")
    net.load_state_dict(state_dict)
    net.eval()

    # Load Data
    data = Data(set_name = "MNIST",
                test_batch_size = 1,
                maxmin=True)
    
    # Load Image
    image, label = next(iter(data.test_loader))

    # Load Optimal U for image
    net.U = torch.load(unitary_root + 'U{}'.format(0))
    UtU = mm(net.U,t(net.U))
    print("UTU:", torch.round(UtU,decimals=0))


    # Ortho Image
