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
    _, eigenvectors = torch.linalg.eig(A)
    eig_min = eigenvectors[-1]
    eig_max = eigenvectors[ 0]
    
    
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

    return V

# Matrix Multiplication
def mm(A, B):
    return torch.mm(A,B)

# Transpose
def t(A):
    return torch.transpose(A, 0, 1)

# Build excel sheet with eigensystem
def build_excel(dict_of_matrices):
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
        borders.top = 5
        borders.top_colour = 0x40b
        borders.bottom = 5
        borders.bottom_colour = 0x40b
        style = xlwt.XFStyle()
        style.borders = borders
        
        
        sheet.write(i*(nrows+4) + 4, ncols + 1, "Eigenvectors", style)
        for j in range(nrows):
            for k in range(ncols):
                sheet.write(i*(nrows+4)+5+j, ncols + 1 + k, round(torch.real(eigenvectors[j,k]).item(),2), style)
                
        # Save        
        workbook.save("EigensystemComparison.xls")



# Main
if __name__ == "__main__":
    # Generate Random Positive Definite
    A = torch.rand(5, 5)
    A = mm(A, t(A))
    A.add_(torch.eye(5))

    # Get unitary
    V = GramSchmidt(A)
    
    # Basis change
    R = torch.eye(5)
    index = torch.tensor(range(R.size(0)))
    index[0:2] = torch.tensor([1, 0])
    R = R[index]
    
    # D = VT A V
    D = mm(mm(t(V), A), V)
    
    # E = RT Q R
    E = mm(mm(t(R), D), R)
    
    # F = V E Vt
    F = mm(mm(V, E), t(V))

    # G = VFTVT A VFVT
    G = mm(mm(mm(mm(mm(mm(V, t(F)), t(V)), A), V), F), t(V))

    # Print to excel
    dict_of_matrices = {"A": A, "V": V, "G": G}
    build_excel(dict_of_matrices)