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
    


# Main
if __name__ == "__main__":
    # Generate Random Positive Definite
    mat_size = 5
    A = torch.rand(mat_size, mat_size)
    A = mm(A, t(A))
    A.add_(torch.eye(mat_size))

    # Orthogonal matrix with first 2 columns as the eigenvectors associated with the min and max eigenvalues
    D = GramSchmidt(A)
  
    # Permutation matrix
    P = torch.eye(mat_size)
    index = torch.tensor(range(P.size(0)))
    index[0:2] = torch.tensor([1, 0])
    P = P[index]

    # Basis change of D from P
    N = mm(mm(D,P),t(D))
    
    # New matrix with 
    B = mm(mm(N,A),t(N))

    