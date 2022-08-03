# Imports 
import torch
from data_setup import Data
from models.classes.first_layer_unitary_net import FstLayUniNet

# Parameters
set_name        = 'MNIST'
unitary_root    = '../../../data/naddeok/mnist_U_files/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/'
gpu             = False
gpu_number      = "6"

if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Load Data
data = Data(set_name = set_name,
            test_batch_size = 1,
            gpu = gpu)
    
net = FstLayUniNet( set_name   = set_name,
                    model_name = "lenet",
                    gpu = gpu)

# Initialize
if set_name == 'MNIST':
    set_size_dict = {   "train" : int(6e4),
                        "test"  : int(1e4)}
    
    num_channels    = 1
    image_width     = 28
    image_height    = 28
    
else:
    print("MNIST is the only set_name currently supported...")
    exit()
    

# Join all images into one dataset
for set_type in set_size_dict.keys():
    # Unload set size from dict
    set_size = set_size_dict[set_type]
    
    if set_type == "train":
        data_loader = data.get_train_loader(batch_size=1, shuffle=False)
    elif set_type == "test":
        data_loader = data.test_loader
    
    # Initialize unitary set
    unitary_images = torch.empty((  set_size,
                                    num_channels, 
                                    image_width, 
                                    image_height))
    
    labels = torch.empty((set_size))

    # Cycle through data set
    for i, (image, label) in enumerate(data_loader):
        
        # Load Optimal U for image
        net.U =  torch.load(unitary_root + set_type + '/U{}'.format(i), 
                            map_location='cuda:0' if gpu else 'cpu')
        
        # Rotate Images
        unitary_image = net.orthogonal_operation(image)
        
        # Store Image
        unitary_images[i,:,:,:] = unitary_image
        labels[i] = label

    # Save Unitary Dataset
    torch.save((unitary_images, labels), unitary_root + set_type + "/" + set_type + 'ing.pt')