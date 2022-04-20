# Imports
import torch
import copy

# Try a unitary transformation that swaps first and last eigenvector
# -> First write a matrix in a basis of the highest and lowest eigenvector than convert to real numbers
# -->> Normalize v_max and v_min, then take space spaned by them, check email from Nidhal


# GramSchmidt Algorithm
def GramSchmidt(A):
    # Get eigensystem
    eigenvalues, eigenvectors = torch.linalg.eig(A)
    eig_max = eigenvectors[ 0]
    eig_min = eigenvectors[-1]
    
    # Generate initial matrix to transform into unitary
    V = torch.randn_like(A)
    V[0,:] = eig_max
    V[1,:] = eig_min

    # Orthogonal complement of V in n-dimension 
    for i in range(A.size(0)):
        # Orthonormalize
        Ui = copy.copy(V[i])

        for j in range(i):
            Uj = copy.copy(V[j])

            
            Ui = Ui - ((torch.dot(Uj.view(-1), Ui.view(-1)) / (torch.linalg.norm(Uj, ord = 2)**2))*Uj)
            
        V[i] = Ui / torch.linalg.norm(Ui, ord = 2)

    return V

# Main
if __name__ == "__main__":
    # Generate Random Positive Definite
    A = torch.rand(5, 5)
    A = torch.mm(A, A.t())
    A.add_(torch.eye(5))

    # Get unitary
    V = GramSchmidt(A)
    print("V", V)
    print("VTV")
    print(print(torch.mm(torch.transpose(V, 0, 1), V)))
    
    # Basis change
    U = torch.eye(5)
    index = torch.tensor(range(U.size(0)))
    index[0] = 1
    index[1] = 0
    U = U[index]
    print("U_rot", U)
    exit()
    
    U = torch.mm(torch.mm(torch.transpose(V, 0, 1), U), V)
    
    print("U", U)
    print("UTU")
    print(torch.mm(torch.transpose(U, 0, 1), U))
    
    B = torch.mm(torch.mm(torch.transpose(U, 0, 1), A), U)
    Aval, Avec = torch.linalg.eig(A)
    Bval, Bvec = torch.linalg.eig(B)

    print("V")
    Vval, Vvec = torch.linalg.eig(V)
    print(Vval.real, Vvec.real)