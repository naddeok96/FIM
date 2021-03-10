# Imports
import torch
from torch import nn
import pytorch_lightning as pl
import torchvision
import pickle
import torch.nn.functional as F
from copy import copy
from unorm import UnNormalize
import torchvision.transforms as transforms
from pytorch_lightning.loggers import WandbLogger

class LitVGG16(pl.LightningModule):

    def __init__(self, U = None,
                       batch_size = 124,
                       learning_rate = 0.001, 
                       momentum = 0.9, 
                       weight_decay = 0.0001,
                       pretrained = True):

        super(LitVGG16, self).__init__()

        self.U = U
        self.batch_size = batch_size
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.criterion = torch.nn.CrossEntropyLoss()

        self.accuracy = pl.metrics.Accuracy()

        self.vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        self.image_size = 3*224*224


    def prepare_data(self):
        
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        self.inverse_transform = UnNormalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        self.train_set = torchvision.datasets.ImageFolder(root='../../../data/ImageNet/train', # '../data' 
                                                transform=self.transform)

        # Train/Val split
        self.train_set, self.val_set = torch.utils.data.random_split(self.train_set, [1174404, 106763])
        
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
        
        return loss

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
        U = copy(U.view((1, self.image_size, self.image_size)).repeat(batch_size, 1, 1))
        
        # Batch muiltply UA
        return torch.bmm(U, input_tensor.view(batch_size, self.image_size, 1)).view(batch_size, channel_num, A_side_size, A_side_size)

    def forward(self, x):
        
        # Unitary transformation
        x = self.orthogonal_operation(x)

        # Feedforward
        x = self.vgg16(x)

        return x