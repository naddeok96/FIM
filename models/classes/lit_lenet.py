# Imports
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision
import pickle
import torch.nn.functional as F
from copy import copy
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger

class LitLeNet(pl.LightningModule):

    def __init__(self, set_name,
                       U = None,
                       batch_size = 124,
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
            # Image Size
            self.image_size = 3*32*32
            self.num_classes = 10

            # Input (3,32,32)
            # Layer 1
            self.conv1 = nn.Conv2d(3, # Input channels
            
                                self.num_kernels_layer1, # Output Channel 
                                kernel_size = 5, 
                                stride = 1, 
                                padding = 0) # Output = (3,28,28)

        elif self.set_name == "MNIST":
            # Image Size
            self.image_size = 1*28*28
            self.num_classes = 10

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
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]) #  Normalize about 0.5

            self.train_set = torchvision.datasets.CIFAR10(root='../../../data/pytorch', # '../data' 
                                                    train=True,
                                                    download=False,
                                                    transform=self.transform)

            self.test_set = torchvision.datasets.CIFAR10(root='../../../data/pytorch',
                                                    train=False,
                                                    download=False,
                                                    transform=self.transform)
            # Train/Val Split
            self.test_set, self.val_set = torch.utils.data.random_split(self.test_set, [9000, 1000])
            
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

            # Train/Val Split
            self.train_set, self.val_set = torch.utils.data.random_split(self.train_set, [55000, 5000])

            self.test_set = torchvision.datasets.MNIST(root='../../../data/pytorch',
                                                    train=False,
                                                    download=False,
                                                    transform=self.transform)
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

        # Log to WandB
        accuracy = self.accuracy(preds, labels)
        self.log('train_acc', accuracy, prog_bar=True, sync_dist=True)
        self.log('train_loss', loss, sync_dist=True)

        output = {
                    'loss': loss, 
                    'prog': {'acc': accuracy}
        }
        return output

    def validation_step(self, batch, batch_idx):
        # Load batch data
        inputs, labels = batch

        # Forward pass
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)

        # Calculate loss
        loss = self.criterion(outputs, labels)

        # Log to WandB
        accuracy = self.accuracy(preds, labels)
        self.log('val_acc', accuracy, prog_bar=True, sync_dist=True)
        self.log('val_loss', loss, sync_dist=True)

        output = {
            'loss': loss, 
            'prog': {'acc': accuracy}
        }
        return output

    def test_step(self, batch, batch_idx):
        # Load batch data
        inputs, labels = batch

        # Forward pass
        outputs = self.forward(inputs)
        preds = torch.argmax(outputs, dim=1)

        # Calculate loss
        loss = self.criterion(outputs, labels)

        # Log to WandB
        accuracy = self.accuracy(preds, labels)
        self.log('test_acc', accuracy, sync_dist=True)
        self.log('test_loss', loss, sync_dist=True)

        output = {
            'loss': loss, 
            'prog': {'acc': accuracy}
        }
        return output

    # Generate orthoganal matrix
    def get_orthogonal_matrix(self, size):
        '''
        Generates an orthoganal matrix of input size
        '''
        # Calculate an orthoganal matrix the size of A
        return torch.nn.init.orthogonal_(torch.empty(size, size))

    def save_orthogonal_matrix(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self.U, output, pickle.HIGHEST_PROTOCOL)

    def load_orthogonal_matrix(self, filename):
        with open(filename, 'rb') as input:
            self.U = pickle.load(input).type(torch.FloatTensor)

     # Set a new U
    def set_orthogonal_matrix(self):
        self.U = self.get_orthogonal_matrix(self.image_size**2)

    def set_random_matrix(self):
        self.U = torch.rand(self.image_size**2, self.image_size**2)

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
            return input_tensor
        else:
            U = copy(self.U)

        # Push to GPU if True
        U = copy(U.cuda())

        # Repeat U and U transpose for all batches
        input_tensor = input_tensor.cuda()
        U = copy(U.view((1, channel_num * A_side_size**2, channel_num * A_side_size**2)).repeat(batch_size, 1, 1))
        
        # Batch muiltply UA
        return torch.bmm(U, input_tensor.view(batch_size, channel_num *A_side_size**2, 1)).view(batch_size, channel_num, A_side_size, A_side_size)

    def forward(self, x):
        
        # Unitary transformation
        x = self.orthogonal_operation(x)

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