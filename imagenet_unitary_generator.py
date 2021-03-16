# Imports 
import torchvision.transforms as transforms
import pickle
import torch
import os

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


# Declare Transformations
size = 3*224*224
U = torch.nn.init.orthogonal_(torch.empty(size, size))
with open("models/pretrained/ImageNet_U.pkl", 'wb') as output:
    pickle.dump(U, output, pickle.HIGHEST_PROTOCOL)
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