

# Imports
import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class PertData:
    def __init__(self,  set_name     = "MNIST",
                        root = '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/',
                        gpu                 = False,
                        test_batch_size     = 256,
                        desired_image_size  = None,
                        maxmin              = False):

        super(PertData,self).__init__()

        import time
        # Hyperparameters
        self.set_name           = set_name
        self.gpu                = gpu
        self.root       = root
        self.test_batch_size    = test_batch_size
        self.desired_image_size = desired_image_size

        # Pull in data
        if self.set_name == "CIFAR10":
            assert self.set_name == "CIFAR10", "CIFAR10 is not setup for PertData yet."
            
            # Set Root
            self.root = '../../../data/pytorch/' if root is None else root

            # Images are of size (3,32,32)
            self.num_classes = 10
            self.num_channels = 3
            self.image_size = 32 if self.desired_image_size is None else self.desired_image_size

            self.mean = (0.4914, 0.4822, 0.4465)
            self.std  = (0.2023, 0.1994, 0.2010)

            # Train Set
            self.set_train_transform()
            self.train_set =  torchvision.datasets.CIFAR10(root=self.root, 
                                                    train=True,
                                                    download=True,
                                                    transform=self.train_transform)

            # Test Set
            self.test_transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                    transforms.ToTensor(), # Convert the images into tensors
                                                    transforms.Normalize(self.mean, 
                                                                        self.std)]) #  Normalize

            self.inverse_transform = UnNormalize(mean=self.mean,
                                                std=self.std)

            self.test_set = torchvision.datasets.CIFAR10(root=self.root,
                                                    train=False,
                                                    download=True,
                                                    transform=self.test_transform)
                                                    
        elif self.set_name == "MNIST":

            # Image size
            self.num_classes = 10
            self.num_channels = 1
            self.image_size = 28 if self.desired_image_size is None else self.desired_image_size

            # Declare Transforms
            self.test_transform = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),
                                                    transforms.ToTensor() # Convert the images into tensors
                                                    ])
                                                    
            self.test_set = UnitaryDataset(unitary_root  = self.unitary_root  + 'test/testing.pt')

        elif self.set_name == "TinyImageNet":
            assert self.set_name == "TinyImageNet", "TinyImageNet is not setup for UnitaryData yet."
            
            # Set Root
            self.root = '../../../data/tiny-imagenet-200/' if root is None else root

            self.mean = (0.4802, 0.4481, 0.3975)
            self.std  = (0.2762, 0.2686, 0.2813)

            # Image size
            self.num_classes  = 200
            self.num_channels = 3
            self.image_size   = 64

            # Declare Transforms
            self.set_train_transform()

            self.test_transform = transforms.Compose([  transforms.ToTensor(), # Convert the images into tensors
                                                        transforms.Normalize(self.mean, self.std)]) # Normalize 

            self.inverse_transform = UnNormalize(mean = self.mean,
                                                    std  = self.std)

            # Generate Datasets
            self.train_set = datasets.ImageFolder(root      = '../../../data/tiny-imagenet-200/' + 'train',
                                                    transform = self.train_transform)

            self.test_set = datasets.ImageFolder(root      = '../../../data/tiny-imagenet-200/' + 'train',
                                                    transform = self.train_transform)

        elif self.set_name == "ImageNet":
            assert self.set_name == "ImageNet", "ImageNet is not setup for UnitaryData yet."
            
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
        if isinstance(self.gpu, bool):

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
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_set,
                                                                                num_replicas=torch.cuda.device_count(), 
                                                                                rank=gpu, 
                                                                                shuffle=False)
            self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                            batch_size = self.test_batch_size,
                                                            num_workers = 8,
                                                            pin_memory = True,
                                                            sampler = test_sampler)
            
        # Determine test sets min/max pixel values
        if maxmin:
            assert not maxmin, "Maxmin is not supported yet"
            self.set_testset_min_max()
            
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

class PertDataset(Dataset):
    """Dataset."""

    def __init__(self, 
                    root    = '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/training.pt', 
                    transforms      = None
                ):        
        
        # Load unitary images
        self.images, self.labels = torch.load(root)
            
        # Store transforms
        self.transforms = transforms

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, idx):
        
        # Ensure proper format
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load image and label
        image, label = self.images[idx,:,:,:], int(self.labels[idx])
        
        # Disable gradient
        image = image.detach()

        # Perform transform
        if self.transforms:
            print("Transform is not properly setup for data augmentation.")
            image = self.transforms(image)
            
        return image, label
