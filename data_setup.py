'''
This class loads the data data and splits it into training and testing sets
'''

# Imports
import torch
from torch import nn
import torchvision
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from unorm import UnNormalize
import matplotlib.pyplot as plt

class Data:
    def __init__(self,set_name = "MNIST",
                      gpu = False,
                      test_batch_size = 256,
                      desired_image_size = None,
                      data_augment = False,
                      maxmin = False):

        super(Data,self).__init__()

        # Hyperparameters
        self.gpu = gpu
        self.set_name = set_name
        self.test_batch_size = test_batch_size
        self.desired_image_size = desired_image_size
        self.data_augment = data_augment

        # Pull in data
        if self.set_name == "CIFAR10":
            # Images are of size (3,32,32)
            self.num_classes = 10
            self.num_channels = 3
            self.image_size = 32 if self.desired_image_size is None else self.desired_image_size

            # Train Set
            self.train_set = self.get_trainset()

            # Test Set
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std  = (0.2023, 0.1994, 0.2010)
            self.test_transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                      transforms.ToTensor(), # Convert the images into tensors
                                                      transforms.Normalize(self.mean, 
                                                                           self.std)]) #  Normalize

            self.inverse_transform = UnNormalize(mean=self.mean,
                                                 std=self.std)

            self.test_set = torchvision.datasets.CIFAR10(root='../data', # '../../../data/pytorch/CIFAR10', #
                                                    train=False,
                                                    download=True,
                                                    transform=self.test_transform)
                                                    
        elif self.set_name == "MNIST":
            # Image size
            self.num_channels = 1
            self.image_size = 28

            # Images are of size (1, 28, 28)
            self.mean = 0.1307
            self.std = 0.3081
            self.train_transform = transforms.Compose([transforms.ToTensor(), # Convert the images into tensors
                                                #  transforms.Normalize((self.mean,), (self.std,))
                                                 ]) # Normalize 

            self.test_transform = transforms.Compose([transforms.ToTensor(), # Convert the images into tensors
                                                 transforms.Normalize((self.mean,), (self.std,))]) # Normalize 

            self.inverse_transform = transforms.Compose([transforms.ToTensor(), 
                                                         transforms.Normalize((-self.mean * self.std,), (1/self.std,))])

            self.train_set = datasets.MNIST(root='../data/',
                                            train = True,
                                            download = True,
                                            transform = self.train_transform) 

            self.test_set = torchvision.datasets.MNIST(root='../data/',
                                                    train=False,
                                                    download=True,
                                                    transform=self.test_transform)

        elif self.set_name == "ImageNet":
            self.train_transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])

            self.test_transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])

            self.inverse_transform = UnNormalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])

            self.train_set = torchvision.datasets.ImageFolder(root='../../../data/ImageNet/train', # '../data' 
                                                    transform=self.train_transform)

            self.test_set = torchvision.datasets.ImageFolder(root='../../../data/ImageNet/val',
                                                    transform=self.test_transform)

            # import matplotlib
            # matplotlib.use('Agg')
            # import matplotlib.pyplot as plt
            # img = torchvision.utils.make_grid((inputs[0]))
            # img = self.data.inverse_transform(img)
            # plt.imshow(np.transpose(img.cpu().numpy(), (1 , 2 , 0)))
            # plt.savefig("img2.png")
            # exit()

        else:
            print("Please enter vaild dataset.")
            exit()

        #Test and validation loaders have constant batch sizes, so we can define them 
        if self.gpu:
            self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                           batch_size = self.test_batch_size,
                                                           shuffle = False,
                                                           num_workers = 8,
                                                           pin_memory = True)
        else:
            self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                           batch_size = self.test_batch_size,
                                                           shuffle = False)

        # Determine test sets min/max pixel values
        if maxmin:
           self.set_testset_min_max()
            
    def get_trainset(self):
        if self.data_augment:
            self.train_transform = transforms.Compose([ transforms.Resize((self.image_size, self.image_size)),
                                                        transforms.RandomRotation(10),
                                                        transforms.RandomAffine(degrees=2, 
                                                                                translate=(0.02, 0.02), 
                                                                                scale=(0.98, 1.02), 
                                                                                shear=2, 
                                                                                fillcolor=(124,117,104)),
                                                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(), # Convert the images into tensors
                                                        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                             (0.2023, 0.1994, 0.2010))]) #  Normalize

        else:
            self.train_transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                      transforms.ToTensor(), # Convert the images into tensors
                                                      transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                           (0.2023, 0.1994, 0.2010))
                                                                           ]) #  Normalize

        return torchvision.datasets.CIFAR10(root='../data', # '../../../data/pytorch/CIFAR10', # 
                                                    train=True,
                                                    download=True,
                                                    transform=self.train_transform)

    # Fucntion to break training set into batches
    def get_train_loader(self, batch_size):
        '''
        Load the train loader given batch_size
        '''
        if self.gpu ==False:
            train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size = batch_size,
                                                        shuffle = True)
        else:
            train_loader = torch.utils.data.DataLoader(self.train_set,
                                                        batch_size = batch_size,
                                                        shuffle = True,
                                                        num_workers = 8,
                                                        pin_memory=True)
        return train_loader

    def set_testset_min_max(self):
        set_min = 1e6
        set_max = 1e-6
        for images, _ in self.test_loader:
            batch_min = torch.min(images.view(-1))
            batch_max = torch.max(images.view(-1))

            if batch_min < set_min:
                set_min = batch_min

            if batch_max > set_max:
                set_max = batch_max

        self.test_pixel_min = set_min
        self.test_pixel_max = set_max

    def unload(self, image_size):
        '''
        Change dimensions of image from [1,1,pixel_size,pixel_size] to [pixel_size, pixel_size]
        '''
        return image_size.squeeze(0).squeeze(0)

    def get_single_image(self, index = "random",
                               show = False):
        '''
        Pulls a single image/label out of test set by index
        '''        
        if index == "random":
            index = torch.randint(high = len(self.test_loader.dataset), size = (1,)).item()

        # Get image
        image = self.test_set.data[index]
        image = image.type('torch.FloatTensor')

        # Get Label
        label = self.test_set.targets[index].item()
        label = torch.tensor([label])

        # Display
        if show:
            fig, ax = plt.subplots()
            fig.suptitle('Label: ' + str(label.item()), fontsize=16)
            plt.xlabel("Index " + str(index))
            plt.imshow(self.unload(image), cmap='gray', vmin = 0, vmax = 255)
            plt.show()

        return image, label, index



