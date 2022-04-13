from models.classes.first_layer_unitary_net  import FstLayUniNet
from adversarial_attacks import Attacker
from data_setup import Data
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision as tv

# Parameters
set_names = ["MNIST", "CIFAR10"]
save_only = True

fig, axes2d = plt.subplots(nrows=2*len(set_names)+1,
                            ncols=10,
                            sharex=True, sharey=True)


for k,set_name in enumerate(set_names):
    # Load Network
    if set_name == "MNIST":
        model_name = "lenet"

    elif set_name == "CIFAR10":
        model_name = "cifar10_mobilenetv2_x1_0"

    net = FstLayUniNet( set_name = set_name,
                        model_name = model_name,
                        desired_image_size = 32)
    net.set_orthogonal_matrix()
    net.eval()

    # Initialize data
    data = Data(set_name = set_name,
                root     = "../data/",
                desired_image_size = 32)


    # Initalize images and labels for one of each number
    images = torch.zeros((data.num_classes, data.num_channels, data.image_size, data.image_size))
    labels = torch.zeros((data.num_classes)).type(torch.LongTensor)

    if set_name == "MNIST":
        label_names = list(range(10))
    else:
        label_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    # Find one of each number
    found = False
    number = 0
    index = 0
    while found is not True:
        for test_inputs, test_labels in data.test_loader:
            
            image, label, = test_inputs[0], test_labels[0]

            if label.item() == number:
                images[number, :, :, :] = image
                labels[number] = label

                number += 1
                if number > 9:
                    found = True
            index += 1

    

    # plt.suptitle(attack + " on " + data.set_name, fontsize=20)
    # fig.text(0.03, 0.5, 'Noise to Signal L2 Norm Ratio', va='center', ha='center', rotation='vertical', fontsize=18)

    for i, row in enumerate(tqdm(axes2d)):
        if k==0 and (i==3 or i==4):
            continue
        elif k==1 and (i==0 or i==1):
            continue
        elif i == 2:
            for j, cell in enumerate(row):
                cell.set_axis_off()
            continue
            

        # Ortho Images
        ortho_images = net.orthogonal_operation(images)

        # # UNnormalize 
        images = images.view(images.size(0), images.size(1), -1)
        batch_means = torch.tensor(data.mean).repeat(images.size(0), 1).view(images.size(0), images.size(1), 1)
        batch_stds  = torch.tensor(data.std).repeat(images.size(0), 1).view(images.size(0), images.size(1), 1)
        images = images.mul_(batch_stds).add_(batch_means)
        images = images.sub_(torch.min(images)).div_(torch.max(images) - torch.min(images)).view(images.size(0), images.size(1), data.image_size, data.image_size)

        ortho_images = ortho_images.view(ortho_images.size(0), ortho_images.size(1), -1)
        batch_means = torch.tensor(data.mean).repeat(ortho_images.size(0), 1).view(ortho_images.size(0), ortho_images.size(1), 1)
        batch_stds  = torch.tensor(data.std).repeat(ortho_images.size(0), 1).view(ortho_images.size(0), ortho_images.size(1), 1)
        ortho_images = ortho_images.mul_(batch_stds).add_(batch_means)
        ortho_images = ortho_images.sub_(torch.min(ortho_images)).div_(torch.max(ortho_images) - torch.min(ortho_images)).view(ortho_images.size(0), ortho_images.size(1), data.image_size, data.image_size)

        for j, cell in enumerate(row):
            # Plot in cell
            img = ortho_images if i==1 or i==4 else images
            img = tv.utils.make_grid(img[j,:,:,:]) 
            cell.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))
            cell.set_xticks([])
            cell.set_yticks([])
            cell.set_axis_off()

            if i == 0 or i == 3:
                cell.set_title(label_names[j])
                    
            # if j == 0:
            #     cell.set_ylabel(set_name)
                
fig.subplots_adjust(hspace = -0.7, wspace=0)
fig.supylabel('CIFAR10        MNIST')
# fig.tight_layout()
plt.axis('off')

    
if save_only:
    plt.savefig('results/MNIST_CIFAR10_Ortho.png')
else:
    print("Display")
    plt.show()


# # Plot
# fig, (ax1, ax2) = plt.subplots(1, 2)

# for i in list(range(2)):
#     if i == 0:
#         img = images
#         ax1.set_xlabel("Original")
#     else:
#         img = ortho_image
#         ax2.set_xlabel("Orthogonally Transformed")

#     # Plot in cell
#     img = tv.utils.make_grid(img)
#     if i == 0:
#         ax1.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))
#         ax1.set_xticks([])
#         ax1.set_yticks([])
#     else:
#         ax2.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))
#         ax2.set_xticks([])
#         ax2.set_yticks([])

         
# fig.subplots_adjust(hspace = -0.4, wspace=0)
# fig.suptitle(set_name, y = 0.82, fontsize = 18)
    
# if save_only:
#     plt.savefig('results/' + data.set_name + '/plots/OrthoImg.png')
# else:
#     plt.show()