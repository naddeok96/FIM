# Imports 
import torchvision.transforms as transforms
import pickle
import torch
import os
import copy
import time

# Decalre GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Generate U
def save_vector(vector_filename, index, vector):
    with open("imagenet_U_files/" + vector_filename + "/" + vector_filename + str(index) + ".pkl", 'wb') as output:
        pickle.dump(vector, output, pickle.HIGHEST_PROTOCOL)

def load_vector(vector_filename, index):
    with open("imagenet_U_files/" + vector_filename + "/" + vector_filename + str(index) + ".pkl", 'rb') as input:
       vector = pickle.load(input).type(torch.FloatTensor)
    return vector

def initalize_random_matrix(mat_filename, size):
    for i in range(size):
        if i % 1000 == 0:
            print("Init " + mat_filename, i)
        random_vector = torch.rand(size, 1)
        save_vector(mat_filename, i, random_vector)

def initalize_identity_matrix(mat_filename, size):
    for i in range(size):
        if i % 1000 == 0:
            print("Init " + mat_filename, i)
        unit_vector = torch.zeros(size)
        unit_vector[i] = 1
        save_vector(mat_filename, i, unit_vector)

def initalize_empty_matrix(mat_filename, size):
    for i in range(size):
        if i % 1000 == 0:
            print("Init " + mat_filename, i)
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

def Householder(size):
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

    return U

def check_U(size):
    for i in range(size):
        Ui = load_vector("U", i)

        if i == 0:
            U = Ui
        else:
            U = torch.cat((U, Ui), dim = 1)

    check = torch.matmul(torch.t(U), U)
    print(torch.round(check))

def vector_by_vector_matmul(transposed_mat1_filename, mat2_filename, out_filename, size):
    
    for i in range(size):
        
        at = load_vector(transposed_mat1_filename, i).view(1, -1)
        print("AT", at)
    
        c = load_vector(out_filename, i).view(-1)

        for j in range(size):
            b = load_vector(mat2_filename, j).view(-1, 1)

            c[j] = torch.matmul(at, b)

        # print("C", c)

        save_vector(out_filename, i, c)
    
def save_transpose(mat_filename, size):

    for i in range(size):
        at = torch.empty(size)

        for j in range(size):
            at[j] = load_vector(mat_filename, j)[i]

        if i % 1000 == 0:
            print("Init " + mat_filename, i)
        save_vector(mat_filename + "T", i, at)

def load_full_matrix(mat_filename, size):
    A = torch.empty(size, size)

    for i in range(size):
        A[:,i] = load_vector(mat_filename, i).view(-1)

    return A

def main():
    # Parameters
    size = 4

    # Initialize Matricies
    initalize_random_matrix("V", size)
    # initalize_identity_matrix("U", size)
    initalize_empty_matrix("Q", size)

    # Matrix Multiply
    save_transpose("V", size)
    vector_by_vector_matmul("VT", "V", "Q", size)

    # Check Step
    V  = load_full_matrix("V", size)
    VT = load_full_matrix("VT", size)
    print("V\n", V)
    print("\nVT\n", VT)

    print("\nVtV\n", torch.matmul(VT, V))
    print("\nQ\n", load_full_matrix("Q", size))

if __name__ == '__main__':
    main()
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