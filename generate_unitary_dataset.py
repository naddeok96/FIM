# Imports 
from operator import index
import torch
import os
import time
import matplotlib.pyplot as plt
from data_setup import Data
from models.classes.first_layer_unitary_net import FstLayUniNet

# Hypers
set_name   = 'MNIST'
start_image_index = 49616
gpu        = True
gpu_number = "3"
batch_size = 1
from_ddp   = True
pretrained_weights_filename = "models/pretrained/MNIST/lenet_w_acc_98.pt"

if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Load Data
data = Data(set_name = set_name,
            test_batch_size = 1,
            gpu = gpu)

# Load Source Network
state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

if from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    
net = FstLayUniNet( set_name   = set_name,
                    model_name = "lenet",
                    gpu = gpu)

train_loader = data.get_train_loader(batch_size=1, shuffle=False)
print("Train Set")
count_down = 0
for i, (image, label) in enumerate(train_loader):
    print("Checking: ", i)
    # Skip
    if i < start_image_index: 
        print("Skip due to start index...")
        continue
    
    # # Give space
    # if count_down > 0:
    #     count_down -= 0
    #     print("Skip due to count down...")
    #     continue
    
    # # Check if done
    # already_done = False
    # for file in os.listdir("../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/"):
    
    #     if 'U{}'.format(i) == file:
    #         already_done = True
            
    #         count_down = 0
    #         break

    # if already_done:
    #     print("Skip due to already done...")
    #     continue
    
    # # Display if made
    # print("Working on: ", i)
    
    if gpu:
        image = image.cuda()
        label = label.cuda()
        
    if torch.sum(torch.isnan(torch.load('../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/U{}'.format(i)))) != 0 or torch.sum(torch.isnan(torch.load('../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/UA{}.pt'.format(i)))) != 0:
        torch.save(image, 'TEMP/U{}'.format(i))
        
        
        # Set a optimal orthogonal matrix
        print("Generating orthogonal matrix")
        net.set_orthogonal_matrix(image)

        # Save Optimal U for Image
        torch.save(net.U, '../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/U{}'.format(i))
        
        # Load Optimal U for image
        # net.U = torch.load('../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/U{}'.format(i))
        
        # Rotate Images
        ortho_image = net.orthogonal_operation(image)
        
        # # Save
        torch.save(ortho_image, '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/UA{}.pt'.format(i))
        
        if torch.isnan(torch.load('../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/train/U{}'.format(i))).any() or torch.isnan(torch.load('../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/train/UA{}.pt'.format(i))).any():
            print("DID NOT FIX")
            break

    
print("Test Set")
for i, (image, label) in enumerate(data.test_loader):
    if gpu:
        image = image.cuda()
        label = label.cuda()
        
    # # Set a optimal orthogonal matrix
    # print("Generating orthogonal matrix")
    # net.set_orthogonal_matrix(image)

    # # Save Optimal U for Image
    # torch.save(net.U, '../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/test/U{}'.format(i))

    # Load Optimal U for image
    net.U = torch.load('../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/test/U{}'.format(i))
    
    # Rotate Images
    ortho_image = net.orthogonal_operation(image)
    
    # Save
    torch.save(ortho_image, '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/UA{}.pt'.format(i))

    
print("donezo")





# # Data Augmentation
# dataset_list = []
# for i in range(20):
#     dataset_list.append(Covid19_CT_Dataset(cts_path[i],masks_path[i],transform1=transforms1['both'], transform2=transforms1['image']))

# dataset_train = torch.utils.data.ConcatDataset(dataset_list[:16])
# dataset_val = torch.utils.data.ConcatDataset(dataset_list[16:])
# print(dataset_train.__len__())
# print(dataset_val.__len__())

# # Save as tensors
# os.mkdir('/content/train_loader')
# os.mkdir('/content/val_loader')
# os.mkdir('/content/train_loader/img')
# os.mkdir('/content/train_loader/mask')
# os.mkdir('/content/val_loader/img')
# os.mkdir('/content/val_loader/mask')

# for i, data in enumerate(dataset_val):
#   torch.save(data[0], '/content/val_loader/img/val_transformed_img{}'.format(i))
#   torch.save(data[1], '/content/val_loader/mask/val_transformed_mask{}'.format(i))

# for i, data in enumerate(dataset_train):
#   torch.save(data[0], '/content/train_loader/img/train_transformed_img{}'.format(i))
#   torch.save(data[1], '/content/train_loader/mask/train_transformed_mask{}'.format(i))
  
  
# # Create Dataset
# class transformed_data(Dataset):
#   def __init__(self, img, mask):
#     self.img = img  #img path
#     self.mask = mask  #mask path
#     self.len = len(os.listdir(self.img))

#   def __getitem__(self, index):
#     ls_img = sorted(os.listdir(self.img))
#     ls_mask = sorted(os.listdir(self.mask))

#     img_file_path = os.path.join(self.img, ls_img[index])
#     img_tensor = torch.load(img_file_path)

#     mask_file_path = os.path.join(self.mask, ls_mask[index])
#     mask_tensor = torch.load(mask_file_path)

#     return img_tensor, mask_tensor

#   def __len__(self):
#     return self.len   

# # Load for model training
# dataset_val = transformed_data('/content/val_loader/img', '/content/val_loader/mask')
# dataset_train = transformed_data('/content/train_loader/img', '/content/train_loader/mask')
# print(dataset_val.__len__())
# print(dataset_train.__len__())
# unet_train_loader = DataLoader(dataset_train, batch_size=10, shuffle=True, num_workers=2)
# unet_val_loader = DataLoader(dataset_val, batch_size=5, shuffle=False, num_workers=2)
# dataset_size = {'train':len(dataset_train), 'val':len(dataset_val)}
# dataloader = {'train':unet_train_loader, 'val':unet_val_loader}

