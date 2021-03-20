"""
This is the code for QR factorization using Householder Transformation.
This program is made in python 3.5.3 but will be compatible to any python 3.4+ version
We used numpy library for matrix manipulation.
Install numpy using ** pip3 install numpy ** command on terminal.
To run the code write ** python3 qr_householder.py ** on terminal
User has to give dimension of the matrix as input in space separated format and matrix will be generated randomly.
QR factorization can be done for both square and non-square matrices and hence the code supports both.
"""
import numpy as np

def column_convertor(x):
    """
    Converts 1d array to column vector
    """
    x.shape = (1, x.shape[0])
    return x

def get_norm(x):
    """
    Returns Norm of vector x
    """
    return np.sqrt(np.sum(np.square(x)))

def householder_transformation(v):
    """
    Returns Householder matrix for vector v
    """
    size_of_v = v.shape[1]
    e1 = np.zeros_like(v)
    e1[0, 0] = 1
    vector = get_norm(v) * e1
    if v[0, 0] < 0:
        vector = - vector
    u = (v + vector).astype(np.float32)
    H = np.identity(size_of_v) - ((2 * np.matmul(np.transpose(u), u)) / np.matmul(u, np.transpose(u)))
    return H

def qr_step_factorization(q, r, iter, n):
    """
    Return Q and R matrices for iter number of iterations.
    """
    v = column_convertor(r[iter:, iter])
    Hbar = householder_transformation(v)
    H = np.identity(n)
    H[iter:, iter:] = Hbar
    r = np.matmul(H, r)
    q = np.matmul(q, H)
    return q, r

def main():
    n, m = list(map(int, input("Write size of the matrix in space separated format\n").split()))
    A = np.random.rand(n,m)
    print('The random matrix is \n', A)
    Q = np.identity(n)
    R = A.astype(np.float32)
    for i in range(min(n, m)):
        # For each iteration, H matrix is calculated for (i+1)th row
        Q, R = qr_step_factorization(Q, R, i, n)
    min_dim = min(m, n)
    R = np.around(R, decimals=6)
    R = R[:min_dim, :min_dim]
    Q = np.around(Q, decimals=6)
    print('A after QR factorization')
    print('R matrix')
    print(R, '\n')
    print('Q matrix')
    print(Q)


if __name__ == "__main__":
    main()