# Imports 
import torchvision.transforms as transforms
import pickle
import torch
import os


# Generate U
def save_vector(UorV, index, vector):
    with open("imagenet_U_files/" + UorV + str(index) + ".pkl", 'wb') as output:
        pickle.dump(vector, output, pickle.HIGHEST_PROTOCOL)

def load_vector(UorV, index):
    with open("imagenet_U_files/" + UorV + str(index) + ".pkl", 'rb') as input:
       vector = pickle.load(input).type(torch.FloatTensor)
    return vector

def initalize_Vs(size):
    for i in range(size):
        random_vector = torch.rand(size, 1)
        save_vector("V", i, random_vector)

def GramSchmidt(size):
    for i in range(size):
        # Load V[:, i] as V
        Vi = load_vector("V", i)

        # Orthonormalize
        if i == 0:
            Ui = Vi / torch.linalg.norm(Vi, ord = 2)
            save_vector("U", i, Ui)

        else:
            Ui = Vi

            for j in range(i):
                Uj = load_vector("U", j)
                Ui = Ui - ((torch.matmul(torch.t(Uj), Ui) / (torch.linalg.norm(Uj, ord = 2)**2))*Uj)

            Ui = Ui / torch.linalg.norm(Ui, ord = 2)
            save_vector("U", i, Ui)

def check_U(size):
    for i in range(size):
        Ui = load_vector("U", i)

        if i == 0:
            U = Ui
        else:
            U = torch.cat((U, Ui), dim = 1)

    check = torch.matmul(torch.t(U), U)
    print(torch.round(check))


size = 10
# initalize_Vs(size)
GramSchmidt(size)
check_U(size)



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