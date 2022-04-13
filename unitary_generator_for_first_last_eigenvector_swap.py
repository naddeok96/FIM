# Imports
import torch
import copy

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

    for i in range(A.size(0)):
        # if i <= 1:
        #     continue

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
    A = torch.rand(5, 5)
    A = torch.mm(A, A.t())
    A.add_(torch.eye(5))

    # Get unitary
    V = GramSchmidt(A)
    print("VTV")
    print(print(torch.mm(torch.transpose(V, 0, 1), V)))
    
    # Basis change
    U = torch.eye(5)
    index = torch.tensor(range(U.size(0)))
    index[0] = 1
    index[1] = 0
    U = U[index]
    U = torch.mm(torch.mm(torch.transpose(V, 0, 1), U), V)

    print("UTU")
    print(torch.mm(torch.transpose(U, 0, 1), U))
    B = torch.mm(torch.mm(torch.transpose(U, 0, 1), A), U)

    print("A")
    Aval, Avec = torch.linalg.eig(A)
    print(Aval.real, Avec.real)
    print("B")
    Bval, Bvec = torch.linalg.eig(B)
    print(Bval.real, Bvec.real)

    print("V")
    Vval, Vvec = torch.linalg.eig(V)
    print(Vval.real, Vvec.real)