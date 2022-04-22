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
    print("Min/Max", torch.dot(eig_min,eig_max))
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
    A = torch.rand(5, 5)
    A = mm(A, t(A))
    # A.add_(torch.eye(5))

    # Get unitary
    D = GramSchmidt(A)
  
    # Basis change
    P = torch.eye(5)
    index = torch.tensor(range(P.size(0)))
    index[0:2] = torch.tensor([1, 0])
    P = P[index]

    N = mm(mm(D,P),t(D))
    
    B = mm(mm(N,A),t(N))
    
    # # D = DT A D
    G = mm(mm(t(D), A), D)
    
    # E = RT Q P
    E = mm(mm(t(P), G), P)
    
    # F = D E Dt
    F = mm(mm(D, E), t(D))

    eigenvaluesA, eigenvectorsA = torch.linalg.eig(A)
    eigenvaluesB, eigenvectorsB = torch.linalg.eig(B)
    min_idxA = torch.argmin(torch.abs(torch.real(eigenvaluesA)))
    max_idxA = torch.argmax(torch.abs(torch.real(eigenvaluesA)))

    min_idxB = torch.argmin(torch.abs(torch.real(eigenvaluesB)))
    max_idxB = torch.argmax(torch.abs(torch.real(eigenvaluesB)))

    print(torch.div(eigenvectorsA[min_idxA], eigenvectorsB[max_idxB]))
    print(torch.div(eigenvectorsA[max_idxA], eigenvectorsB[min_idxB]))


    # Print to excel
    # dict_of_matrices = {"A": A, "D":D, "D":D,"E":E,"F":F}
    dict_of_matrices = {"A": A, "B":B, "F":F}
    build_excel(dict_of_matrices)


