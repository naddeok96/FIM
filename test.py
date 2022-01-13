
import torch

size = 3

# Generate PD Matrix
A = torch.rand(size, size)
A = torch.mm(A, A.t())
A.add_(torch.eye(size))

B = torch.rand(size, size)
B = torch.mm(B, B.t())
B.add_(torch.eye(size))

d = torch.rand(1, size)

print(d.size())
print(torch.mm(d, A).size())

# print(torch.linalg.eigvals(A))

# C = torch.mm(torch.mm(B.t(), A), B)

# print(torch.linalg.eigvals(C))

# Gs = A
# Gx = C
# J  = B

# torch.sum(torch.linalg.eigvals(Gs))
# torch.sum()