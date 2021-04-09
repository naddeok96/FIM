# Imports 
import torchvision.transforms as transforms
import numpy as np
import pickle
import torch
import os
import shutil
import copy
import time

def initalize_dir(folder_path, dir):
    if check_dir_exists(folder_path, dir):
        if not check_dir_empty(folder_path, dir):
            clean_dir(folder_path, dir)
    else:
        os.mkdir(folder_path + dir)

def check_dir_empty(folder_path, dir):
    return (len(os.listdir(folder_path + dir)) == 0)

def check_dir_exists(folder_path, dir):
    return (dir in os.listdir(folder_path))

def clean_dir(folder_path, dir):
    for filename in os.listdir(folder_path + dir + "/"):
        file_path = os.path.join(folder_path + dir + "/", filename)
        os.remove(file_path)

def save_status(folder_path, status):
    f = open(folder_path + "status.txt", "w")
    f.write(str(status))
    f.close()

def save_vector(folder_path, vector_filename, index, vector):
    if not check_dir_exists(folder_path, vector_filename):
        os.mkdir(folder_path + vector_filename)

    with open(folder_path + vector_filename + "/" + vector_filename + str(index) + ".pkl", 'wb') as output:
        pickle.dump(vector, output, pickle.HIGHEST_PROTOCOL)

def save_vector_in_temp_dir(folder_path, vector_filename, index, vector):
    if not check_dir_exists(folder_path, "Temp"):
        os.mkdir(folder_path + "/Temp")

    with open(folder_path + "/Temp/" + vector_filename + str(index) + ".pkl", 'wb') as output:
        pickle.dump(vector, output, pickle.HIGHEST_PROTOCOL)

def load_vector(folder_path, vector_filename, index):
    with open(folder_path + vector_filename + "/" + vector_filename + str(index) + ".pkl", 'rb') as input:
       vector = pickle.load(input).type(torch.FloatTensor)
    return vector.cuda()

def get_saved_column_size(folder_path, vector_filename):
    return len(os.listdir(folder_path + vector_filename + "/"))

def initalize_random_matrix(folder_path, mat_filename, size):
    for i in range(size):
        random_vector = torch.rand(size, 1)
        save_vector(folder_path, mat_filename, i, random_vector)

def initalize_identity_matrix(folder_path, mat_filename, size):
    for i in range(size):
        unit_vector = torch.zeros(size)
        unit_vector[i] = 1
        save_vector(folder_path, mat_filename, i, unit_vector)

def initalize_empty_matrix(folder_path, mat_filename, size):
    for i in range(size):
        empty_vector = torch.empty(size).view(-1, 1)
        save_vector(folder_path, mat_filename, i, empty_vector)

def GramSchmidt(folder_path, size):
    for i in range(size):

        if i < 356: 
            continue

        if i % 10 == 0:
            print("Build U", i)

        # Load V[:, i] as V
        Vi = load_vector(folder_path, "V", i)

        # Orthonormalize
        if i == 0:
            Ui = Vi / torch.linalg.norm(Vi, ord = 2)
            save_vector(folder_path, "U", i, Ui)

        else:
            start = time.time()
            Ui = copy.copy(Vi)
            Ui = Ui.cuda()

            for j in range(i):
                Uj = load_vector(folder_path, "U", j)
                Uj = Uj.cuda()

                
                Ui = Ui - ((torch.dot(Uj.view(-1), Ui.view(-1)) / (torch.linalg.norm(Uj, ord = 2)**2))*Uj)
                
            Ui = Ui / torch.linalg.norm(Ui, ord = 2)

            save_vector(folder_path, "U", i, Ui)

def Householder(folder_path, size, print_every):
    # initalize_dir(folder_path, "V")
    # initalize_random_matrix(folder_path, "V", size)

    # initalize_dir(folder_path, "U")
    # initalize_identity_matrix(folder_path, "U", size)

    print("\nBeginning Householder QR Decomp...\n--------------------------------------------------")
    status = {"Initializing" : "None",
              "Overall Iteration": "Not Started",
              "Hbar Iteration": "Not Started",
              "H Iteration": "Not Started",
              "U" : "Not Started",
              "V" : "Not Started"}
    save_status(folder_path, status)
    start = time.time()
    for i in range(size):
        status["Overall Iteration"] = i
        status["Hbar Iteration"]    = "Not Started for Iteration " + str(i)
        status["H Iteration"]       = "Not Started for Iteration " + str(i)
        save_status(folder_path, status)

        # Load column of V
        V = load_vector(folder_path, "V", i)

        # Calculate x vector
        x = V[i:].view(-1)
        x[0] = x[0] + torch.sign(x[0]) * torch.norm(x, p=2)
        x = x.cuda()

        # Compute H matrix
        if i != 0:
            initalize_dir(folder_path, "Hbar")
            initalize_empty_matrix(folder_path, "Hbar", size - i)
            initalize_dir(folder_path, "H")
            initalize_identity_matrix(folder_path, "H", size)

        status["Hbar Iteration"] = "calculating coefficient for iteration " + str(i)
        save_status(folder_path, status)
        coefficent = 2/torch.dot(x, x)
        status["Hbar Iteration"] = "x by x running for iteration " + str(i)
        save_status(folder_path, status)
        vector_by_vector_dot2mat(folder_path, x, x, "Hbar")
        for j in range(x.view(-1).size(0)):
            status["Hbar Iteration"] = j
            save_status(folder_path, status)
            Hbar =  -coefficent * load_vector(folder_path, "Hbar", j)
            Hbar[j] = Hbar[j] + 1

            save_vector(folder_path, "Hbar", j, Hbar)

        status["Hbar Iteration"] = "Done for Iteration " + str(i)
        save_status(folder_path, status)

        for j, k  in enumerate(range(i, size)):
            status["H Iteration"] = j
            save_status(folder_path, status)
            H = load_vector(folder_path, "H", k)
            Hbar = load_vector(folder_path, "Hbar", j)
            H[i:] = Hbar
            save_vector(folder_path, "H", k, H)   

        status["H Iteration"] = "Done for Iteration " + str(i)    
        save_status(folder_path, status)

        # Clean
        status["Hbar Iteration"] = "Cleaning for iteration " + str(i)
        save_status(folder_path, status)
        shutil.rmtree(folder_path + "/Hbar")
        status["Hbar Iteration"] = "Done cleaning for iteration " + str(i)
        save_status(folder_path, status)

        # Update U and V
        status ["V"] = "Updating for iteration " + str(i)
        save_status(folder_path, status)
        vector_by_vector_matmul(folder_path, "H", "V", "V", size)
        status ["V"] = "Updated for iteration " + str(i)
        status ["U"] = "Updating for iteration " + str(i)
        save_status(folder_path, status)
        vector_by_vector_matmul(folder_path, "U", "H", "U", size)
        status ["U"] = "Updated for iteration " + str(i)
        save_status(folder_path, status)
        
        # Clean
        status["H Iteration"] = "Cleaning for iteration " + str(i)
        save_status(folder_path, status)
        shutil.rmtree(folder_path + "/H")
        status["H Iteration"] = "Done cleaning for iteration " + str(i)
        save_status(folder_path, status)

        # Talk
        if i % print_every == 0 and i != 0:
            print("Column: ", i, " of ", size - 1, "\tTime to make ", print_every, " column(s): ", round(time.time() - start, 2) , " [s]")
            start = time.time()
        elif i == 0:
            print("Time to make first column (usually takes longer than rest to initalize everything): ", round(time.time() - start, 2) , " [s]")
            start = time.time()

    shutil.rmtree(folder_path + "/V")
    status["Overall Iteration"] = "Done"
    status["Hbar Iteration"] ="Done"
    status["H Iteration"] = "Done"
    save_status(folder_path, status)

def check_U(folder_path, size):
    for i in range(size):
        Ui = load_vector(folder_path, "U", i)

        if i == 0:
            U = Ui
        else:
            U = torch.cat((U, Ui), dim = 1)

    check = torch.matmul(torch.t(U), U)
    print(torch.round(check))

def vector_by_vector_dot2mat(folder_path, vec1, vec2, product_mat_filename):
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

        save_vector(folder_path, product_mat_filename, i, c)

def vector_by_vector_matmul(folder_path, mat1_filename, mat2_filename, product_mat_filename, size):
    # Save on Temp File
    initalize_dir(folder_path, "Temp")

    # Save the rows of the first matrix
    save_transpose(folder_path, mat1_filename, size)

    # Determine Dimensions
    ncols = get_saved_column_size(folder_path, mat2_filename)
    nrows = get_saved_column_size(folder_path, mat1_filename + "T")

    # MatMul
    for i in range(ncols):
        # Load Column of B Matrix
        b = load_vector(folder_path, mat2_filename, i).view(-1)
    
        # Initalize Column of product Matrix
        c = load_vector(folder_path, product_mat_filename, i).view(-1)

        for j in range(nrows):
            at = load_vector(folder_path, mat1_filename + "T", j).view(-1)

            c[j] = torch.dot(at, b)

        save_vector_in_temp_dir(folder_path, product_mat_filename, i, c)

    
    shutil.rmtree(folder_path + product_mat_filename)
    os.replace(folder_path + "/Temp", folder_path + product_mat_filename)
    shutil.rmtree(folder_path + mat1_filename + "T")
        
def save_transpose(folder_path, mat_filename, size):
    initalize_dir(folder_path, mat_filename + "T")

    # Decalre dimensions
    nrows = size
    ncols = get_saved_column_size(folder_path, mat_filename)

    # Convert
    for i in range(nrows):
        at = torch.empty(ncols)

        for j in range(ncols):
            at[j] = load_vector(folder_path, mat_filename, j)[i]

        save_vector(folder_path, mat_filename + "T", i, at)

def load_full_matrix(folder_path, mat_filename, size):
    A = torch.empty(size, size)

    for i in range(size):
        A[:,i] = load_vector(folder_path, mat_filename, i).view(-1)

    return A

def main():
    # Decalre GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    # Parameters
    folder_path = "../../../data/naddeok/imagenet_U_files/"
    size = 3*224*224
    print_every = np.ceil(size/100)

    # Generate U Matrix with Householder
    Householder(folder_path, size, print_every)

    # # Load U
    # U = load_full_matrix(folder_path, "U", size)
    # print("U\n", U)
    # check = torch.matmul(torch.t(U), U)
    # print("\nUTU\n", abs(torch.round(check)))

    # Matrix Multiply
    # vector_by_vector_matmul(folder_path, "VT", "V", "Q", size)

    # Check Step
    # V  = load_full_matrix(folder_path, "V", size)
    # VT = load_full_matrix(folder_path, "VT", size)
    # print("V\n", V)
    # print("\n1by1 VT\n", VT)

    # print("\nVV\n", torch.matmul(V, V))
    # print("\nQ\n", load_full_matrix(folder_path, "Q", size))

import time
start = time.time()
if __name__ == '__main__':
    main()

print("--------------------------------------------------\nTotal run time: ", round(time.time() - start, 2), " [s]\n--------------------------------------------------")

# check_U(size)





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