# Imports
import torch

size = 10
V = torch.rand((size, size))
U = torch.eye(size)

for i in range(size):

    # Calculate u vector
    x = V[i:, i].view(-1, 1)
    x[0] = x[0] + torch.sign(x[0]) * torch.norm(x, p=2)

    # Compute H matrix
    Hbar = torch.eye(x.size(0)) - (2 * torch.matmul(x, torch.t(x)) / torch.matmul(torch.t(x), x))
    H = torch.eye(size)
    H[i:, i:] = Hbar

    # Update Q and R
    V = torch.matmul(H, V)
    U = torch.matmul(U, H)

print(torch.round(torch.matmul(torch.transpose(U, 0, 1), U)))

