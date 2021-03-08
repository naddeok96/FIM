# Imports
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from copy import copy
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger

class LitLeNet(pl.LightningModule):

    def __init__(self, set_name,
                       U = None,
                       batch_size = 124,
                       num_classes = 10,
                       num_kernels_layer1 = 6, 
                       num_kernels_layer2 = 16, 
                       num_kernels_layer3 = 120,
                       num_nodes_fc_layer = 84,
                       learning_rate = 0.001, 
                       momentum = 0.9, 
                       weight_decay = 0.0001):

        super(LitLeNet, self).__init__()


        self.set_name = set_name

        self.U = U
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.num_kernels_layer1 = num_kernels_layer1
        self.num_kernels_layer2 = num_kernels_layer2
        self.num_kernels_layer3 = num_kernels_layer3
        self.num_nodes_fc_layer = num_nodes_fc_layer

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.criterion = torch.nn.CrossEntropyLoss()

        self.accuracy = pl.metrics.Accuracy()

        if self.set_name == "CIFAR10":
            self.image_size = 32
            # Input (3,32,32)
            # Layer 1
            self.conv1 = nn.Conv2d(3, # Input channels
            
                                self.num_kernels_layer1, # Output Channel 
                                kernel_size = 5, 
                                stride = 1, 
                                padding = 0) # Output = (3,28,28)

        elif self.set_name == "MNIST":
            self.image_size = 28
            # Input (1,28,28)
            # Layer 1
            self.conv1 = nn.Conv2d(1, # Input channels
                                self.num_kernels_layer1, # Output Channel 
                                kernel_size = 5, 
                                stride = 1, 
                                padding = 2) # Output = (1,28,28)
        else:
            print("Please enter a valid dataset")
            exit()

        # Layer 2
        self.pool1 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 0) # Output = (num_kernels_layer1,14,14)

        # Layer 3
        self.conv2 = nn.Conv2d(self.num_kernels_layer1,
                               self.num_kernels_layer2,
                               kernel_size = 5, 
                               stride = 1, 
                               padding = 0) # Output = (num_kernels_layer2,10,10)

        # Layer 4
        self.pool2 = nn.MaxPool2d(kernel_size = 2, 
                                  stride = 2, 
                                  padding = 0) # Output = (num_kernels_layer3,5,5)

        # Layer 5
        self.conv3 = nn.Conv2d(self.num_kernels_layer2,
                               self.num_kernels_layer3,
                               kernel_size = 5, 
                               stride = 1, 
                               padding = 0) # Output = (num_kernels_layer4,1,1)

        self.fc1 = nn.Linear(self.num_kernels_layer3, self.num_nodes_fc_layer)

        self.fc2 = nn.Linear(self.num_nodes_fc_layer, self.num_classes)


    def prepare_data(self):
        # Pull in data
        if self.set_name == "CIFAR10":
            # Images are of size (3,32,32)
            self.transform = transforms.Compose([transforms.ToTensor(), # Convert the images into tensors
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #  Normalize about 0.5

            self.train_set = torchvision.datasets.CIFAR10(root='../../../data/pytorch', # '../data' 
                                                    train=True,
                                                    download=False,
                                                    transform=self.transform)

            self.test_set = torchvision.datasets.CIFAR10(root='../../../data/pytorch',
                                                    train=False,
                                                    download=False,
                                                    transform=self.transform)
        elif self.set_name == "MNIST":
            # Images are of size (1, 28, 28)
            self.mean = 0.1307
            self.std = 0.3081
            self.transform = transforms.Compose([transforms.ToTensor(), # Convert the images into tensors
                                                 transforms.Normalize((self.mean,), (self.std,))]) # Normalize 
            self.inverse_transform = transforms.Compose([transforms.ToTensor(), 
                                                         transforms.Normalize((-self.mean * self.std,), (1/self.std,))])

            self.train_set = torchvision.datasets.MNIST(root='../../../data/pytorch',
                                            train = True,
                                            download = False,
                                            transform = self.transform) 

            # train/val split
            self.train_set, self.val_set = torch.utils.data.random_split(self.train_set, [55000, 5000])

            self.test_set = torchvision.datasets.MNIST(root='../../../data/pytorch',
                                                    train=False,
                                                    download=False,
                                                    transform=self.transform)

        elif self.set_name == "ImageNet":
            self.transform = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])])

            self.inverse_transform = UnNormalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
            self.train_set = torchvision.datasets.ImageFolder(root='../../../data/ImageNet/train', # '../data' 
                                                    transform=self.transform)

            self.test_set = torchvision.datasets.ImageFolder(root='../../../data/ImageNet/val',
                                                    transform=self.transform)

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

    def train_dataloader(self):
        #Load the dataset
        train_loader = torch.utils.data.DataLoader(self.train_set,
                                                    batch_size = self.batch_size,
                                                    shuffle = True,
                                                    pin_memory=True,
                                                    num_workers=8)                           
        return train_loader

    def val_dataloader(self):
        #Load the dataset
        val_loader = torch.utils.data.DataLoader(self.val_set,
                                                    batch_size = self.batch_size,
                                                    shuffle = False,
                                                    pin_memory=True,
                                                    num_workers=8)      
        return val_loader

    def test_dataloader(self):
        #Load test data
        test_loader = torch.utils.data.DataLoader(self.test_set,
                                                    batch_size = self.batch_size,
                                                    shuffle = False,
                                                    pin_memory=True,
                                                    num_workers=8)
        return test_loader

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), 
                                lr = self.learning_rate, 
                                momentum = self.momentum,
                                weight_decay = self.weight_decay)

    def training_step(self, batch, batch_idx):
        # Load batch data
        inputs, labels = batch

        # Forward pass
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)

        # Calculate loss
        loss = self.criterion(outputs, labels)

        # Using TrainResult to enable logging
        self.log('train_acc', self.accuracy(preds, labels), sync_dist=True)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Load batch data
        inputs, labels = batch

        # Forward pass
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)

        # Calculate loss
        loss = self.criterion(outputs, labels)

        # Using TrainResult to enable logging
        self.log('val_acc', self.accuracy(preds, labels), sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Load batch data
        inputs, labels = batch

        # Forward pass
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)

        # Calculate loss
        loss = self.criterion(outputs, labels)

        # Using TrainResult to enable logging
        accuracy = self.accuracy(preds, labels)
        self.log('test_acc', accuracy, prog_bar=True, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)

    # Orthogonal transformation
    def orthogonal_operation(self, input_tensor):
        '''
        input tensor A nxn
        generate orthoganal matrix U the size of (n*n)x(n*n)

        Returns UA
        '''
        # Find batch size and feature map size
        batch_size = input_tensor.size(0)
        channel_num = int(input_tensor.size(1))
        A_side_size = int(input_tensor.size(2))

        # Determine if U is available
        if self.U == None:
            U = copy(torch.eye(A_side_size**2))
        else:
            U = copy(self.U)

        # Push to GPU if True
        U = copy(U.cuda())

        # Repeat U and U transpose for all batches
        input_tensor = input_tensor.cuda()
        U = copy(U.view((1, A_side_size**2, A_side_size**2)).repeat(channel_num * batch_size, 1, 1))
        
        # Batch muiltply UA
        return torch.bmm(U, input_tensor.view(channel_num *batch_size, A_side_size**2, 1)).view(batch_size, channel_num, A_side_size, A_side_size)

    def forward(self, x):
        
        # Unitary transformation
        # x = self.orthogonal_operation(x)

        # Feedforward
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = x.view(-1, self.num_kernels_layer3 * 1 * 1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x