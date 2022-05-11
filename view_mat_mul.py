
import numpy as np
import xlwt

size = 3

# Generate matrix and vector form
mat = []
vec = []
for i in range(size):
    mat.append([])
    for j in range(size):
        element = "a_{" + str(i+1) + str(j+1) + "}"
        mat[i].append(element)
        vec.append(element)

mat = np.matrix(mat)
vec = np.transpose(np.matrix(vec))




def matmul(a,b):
    c = []
    for i in range(np.shape(a)[0]): # Rows of A
        c.append([])
        for j in range(np.shape(b)[0]): # Cols of B
            element = ''
            for k in range(np.shape(a)[1]): # Cols of A
                if k != 0:
                    element = element + "+"
                
                element = element + 'a_{'+ str(i+1) + str(k+1) + '}b_{' + str(k+1) + str(j+1) + '}' 

            
            c[i].append(element)
    
    return np.matrix(c)

mat_prod = matmul(mat, mat)
vec_prod = matmul(vec,vec)

# print(mat_prod)

def build_excel(A, B):

    workbook = xlwt.Workbook() 
    sheet = workbook.add_sheet("MatMul")

    for i in range(np.shape(A)[0]):
        for j in range(np.shape(A)[1]):


            sheet.write(i, j, A[i,j])

    for i in range(np.shape(B)[0]):
        for j in range(np.shape(B)[1]):


            sheet.write(i, np.shape(A)[0] + 2 + j, B[i,j])

    workbook.save("MatMul.xls")

build_excel(mat_prod, vec_prod)