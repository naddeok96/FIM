# Imports 
import torchvision.transforms as transforms
import pickle
import torch
import os
import shutil
import copy
import time

def initalize_dir(dir):
    if check_dir_exists(dir):
        if not check_dir_empty(dir):
            clean_dir(dir)
    else:
        os.mkdir("../../../data/naddeok/imagenet_U_files/" + dir)

def check_dir_empty(dir):
    return (len(os.listdir("../../../data/naddeok/imagenet_U_files/" + dir)) == 0)

def check_dir_exists(dir):
    return (dir in os.listdir("../../../data/naddeok/imagenet_U_files/"))

def clean_dir(dir):
    for filename in os.listdir("../../../data/naddeok/imagenet_U_files/" + dir + "/"):
        file_path = os.path.join("../../../data/naddeok/imagenet_U_files/" + dir + "/", filename)
        os.remove(file_path)

def save_vector(vector_filename, index, vector):
    if not check_dir_exists(vector_filename):
        os.mkdir("../../../data/naddeok/imagenet_U_files/" + vector_filename)

    with open("../../../data/naddeok/imagenet_U_files/" + vector_filename + "/" + vector_filename + str(index) + ".pkl", 'wb') as output:
        pickle.dump(vector, output, pickle.HIGHEST_PROTOCOL)

def save_vector_in_temp_dir(vector_filename, index, vector):
    if not check_dir_exists("Temp"):
        os.mkdir("../../../data/naddeok/imagenet_U_files/Temp")

    with open("../../../data/naddeok/imagenet_U_files/Temp/" + vector_filename + str(index) + ".pkl", 'wb') as output:
        pickle.dump(vector, output, pickle.HIGHEST_PROTOCOL)

def load_vector(vector_filename, index):
    with open("../../../data/naddeok/imagenet_U_files/" + vector_filename + "/" + vector_filename + str(index) + ".pkl", 'rb') as input:
       vector = pickle.load(input).type(torch.FloatTensor)
    return vector

def get_saved_column_size(vector_filename):
    return len(os.listdir("../../../data/naddeok/imagenet_U_files/" + vector_filename + "/"))

def initalize_random_matrix(mat_filename, size):
    for i in range(size):
        random_vector = torch.rand(size, 1)
        save_vector(mat_filename, i, random_vector)

def initalize_identity_matrix(mat_filename, size):
    for i in range(size):
        unit_vector = torch.zeros(size)
        unit_vector[i] = 1
        save_vector(mat_filename, i, unit_vector)

def initalize_empty_matrix(mat_filename, size):
    for i in range(size):
        empty_vector = torch.empty(size).view(-1, 1)
        save_vector(mat_filename, i, empty_vector)

def GramSchmidt(size):
    for i in range(size):

        if i < 356: 
            continue

        if i % 10 == 0:
            print("Build U", i)

        # Load V[:, i] as V
        Vi = load_vector("V", i)

        # Orthonormalize
        if i == 0:
            Ui = Vi / torch.linalg.norm(Vi, ord = 2)
            save_vector("U", i, Ui)

        else:
            start = time.time()
            Ui = copy.copy(Vi)
            Ui = Ui.cuda()

            for j in range(i):
                Uj = load_vector("U", j)
                Uj = Uj.cuda()

                
                Ui = Ui - ((torch.dot(Uj.view(-1), Ui.view(-1)) / (torch.linalg.norm(Uj, ord = 2)**2))*Uj)
                
            Ui = Ui / torch.linalg.norm(Ui, ord = 2)

            save_vector("U", i, Ui)

def Householder(size, print_every):
    initalize_dir("V")
    initalize_random_matrix("V", size)

    initalize_dir("U")
    initalize_identity_matrix("U", size)

    print("Beginning Householder QR Decomp...")
    start = time.time()
    for i in range(size):
        # Load column of V
        V = load_vector("V", i)

        # Calculate x vector
        x = V[i:].view(-1)
        x[0] = x[0] + torch.sign(x[0]) * torch.norm(x, p=2)

        # Compute H matrix
        initalize_dir("Hbar")
        initalize_empty_matrix("Hbar", size - i)
        initalize_dir("H")
        initalize_identity_matrix("H", size)

        coefficent = 2/torch.dot(x, x)
        vector_by_vector_dot2mat(x, x, "Hbar")
        for j in range(x.view(-1).size(0)):
            Hbar =  -coefficent * load_vector("Hbar", j)
            Hbar[j] = Hbar[j] + 1

            save_vector("Hbar", j, Hbar)
        
        for j, k  in enumerate(range(i, size)):
            H = load_vector("H", k)
            Hbar = load_vector("Hbar", j)
            H[i:] = Hbar
            save_vector("H", k, H)       

        # Clean
        shutil.rmtree("../../../data/naddeok/imagenet_U_files/Hbar")

        # Update U and V
        vector_by_vector_matmul("H", "V", "V", size)
        vector_by_vector_matmul("U", "H", "U", size)
        
        # Clean
        shutil.rmtree("../../../data/naddeok/imagenet_U_files/H")

        # Talk
        if i % print_every == 0:
            print("Column: ", i, " of ", size, "\tTime to make ", print_every, " column(s): ", round(time.time() - start, 2) , " [s]")
            start = time.time()
    shutil.rmtree("../../../data/naddeok/imagenet_U_files/V")

def check_U(size):
    for i in range(size):
        Ui = load_vector("U", i)

        if i == 0:
            U = Ui
        else:
            U = torch.cat((U, Ui), dim = 1)

    check = torch.matmul(torch.t(U), U)
    print(torch.round(check))

def vector_by_vector_dot2mat(vec1, vec2, product_mat_filename):
    # Correct the size
    vec1 = vec1.view(-1)
    vec2 = vec2.view(-1)

    # Determine Dimensions
    ncols = vec1.size(0)
    nrows = vec2.size(0)

    # MatMul
    for i in range(ncols):
        # Load column of vector 2
        b = vec2[i]
    
        # Initalize Column of product Matrix
        c = torch.empty(nrows)

        for j in range(nrows):
            # Load row of vector 1
            at = vec1[j]
            
            c[j] = at*b

        save_vector(product_mat_filename, i, c)

def vector_by_vector_matmul(mat1_filename, mat2_filename, product_mat_filename, size):
    # Save on Temp File
    initalize_dir("Temp")

    # Save the rows of the first matrix
    save_transpose(mat1_filename, size)

    # Determine Dimensions
    ncols = get_saved_column_size(mat2_filename)
    nrows = get_saved_column_size(mat1_filename + "T")

    # MatMul
    for i in range(ncols):
        # Load Column of B Matrix
        b = load_vector(mat2_filename, i).view(-1)
    
        # Initalize Column of product Matrix
        c = load_vector(product_mat_filename, i).view(-1)

        for j in range(nrows):
            at = load_vector(mat1_filename + "T", j).view(-1)

            c[j] = torch.dot(at, b)

        save_vector_in_temp_dir(product_mat_filename, i, c)

    
    shutil.rmtree("../../../data/naddeok/imagenet_U_files/" + product_mat_filename)
    os.replace("../../../data/naddeok/imagenet_U_files/Temp", "../../../data/naddeok/imagenet_U_files/" + product_mat_filename)
    shutil.rmtree("../../../data/naddeok/imagenet_U_files/" + mat1_filename + "T")
        
def save_transpose(mat_filename, size):
    initalize_dir(mat_filename + "T")

    # Decalre dimensions
    nrows = size
    ncols = get_saved_column_size(mat_filename)

    # Convert
    for i in range(nrows):
        at = torch.empty(ncols)

        for j in range(ncols):
            at[j] = load_vector(mat_filename, j)[i]

        save_vector(mat_filename + "T", i, at)

def load_full_matrix(mat_filename, size):
    A = torch.empty(size, size)

    for i in range(size):
        A[:,i] = load_vector(mat_filename, i).view(-1)

    return A

def main():
    # Parameters
    size = 4
    print_every = 1

    # Generate U Matrix with Householder
    Householder(size, print_every)

    # Load U
    # U = load_full_matrix("U", size)
    # print("U\n", U)
    # check = torch.matmul(torch.t(U), U)
    # print("\nUTU\n", abs(torch.round(check)))

    # Matrix Multiply
    # vector_by_vector_matmul("VT", "V", "Q", size)

    # Check Step
    # V  = load_full_matrix("V", size)
    # VT = load_full_matrix("VT", size)
    # print("V\n", V)
    # print("\n1by1 VT\n", VT)

    # print("\nVV\n", torch.matmul(V, V))
    # print("\nQ\n", load_full_matrix("Q", size))

import time
start = time.time()
if __name__ == '__main__':
    main()

print(time.time() - start)
# check_U(size)



# Decalre GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Generate Unitary Transform
# class UnitaryTransform(object):

#     def __init__(self, U):
#         self.U

#     def __call__(self, pic):
#         print(pic.size())
#         exit()
#         return torch.mm(self.U, input_tensor.view(-1, 1))

#     def __repr__(self):
#         return self.__class__.__name__ + '()'




# U = torch.nn.init.orthogonal_(torch.empty(size, size))
# with open("models/pretrained/ImageNet_U.pkl", 'wb') as output:
#     pickle.dump(U, output, pickle.HIGHEST_PROTOCOL)
# transform = transforms.Compose([transforms.Resize((224, 224)),
#                                 transforms.RandomHorizontalFlip(),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                         std=[0.229, 0.224, 0.225]),
#                                 UnitaryTransform(U)])

# Get Train Set  
# train_set = torchvision.datasets.ImageFolder(root='../../../data/ImageNet/train', # '../data' 
#                                         transform=transform)

# # Train/Val split
# self.train_set, self.val_set = torch.utils.data.random_split(self.train_set, [1174404, 106763])

# self.test_set = torchvision.datasets.ImageFolder(root='../../../data/ImageNet/val',
#                                         transform=self.transform)