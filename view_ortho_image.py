from models.classes.first_layer_unitary_net  import FstLayUniNet
from adversarial_attacks import Attacker
from data_setup import Data
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision as tv

# Parameters
set_name    = "CIFAR10"
save_only   = True

# Load Network
if set_name == "MNIST":
    model_name = "lenet"
    U_filename = "models/pretrained/MNIST/U_w_means_0-10024631768465042_and_stds_0-9899614453315735_.pt"

elif set_name == "CIFAR10":
    model_name = "cifar10_mobilenetv2_x1_0"
    U_filename = "models/pretrained/CIFAR10/U_w_means_0-005174736492335796_n0-0014449692098423839_n0-0010137659264728427_and_stds_1-130435824394226_1-128873586654663_1-1922636032104492_.pt"
    
net = FstLayUniNet( set_name = set_name,
                    U_filename = U_filename,
                    model_name = model_name)
net.eval()

# Initialize data
data = Data(set_name = set_name,
            root     = "../data/",
            desired_image_size=32)

# Load Images
image, _ = next(iter(data.get_train_loader(batch_size = 1)))
ortho_image = net.orthogonal_operation(image)

# UNnormalize 
image = image.view(image.size(0), image.size(1), -1)
batch_means = torch.tensor(data.mean).repeat(image.size(0), 1).view(image.size(0), image.size(1), 1)
batch_stds  = torch.tensor(data.std).repeat(image.size(0), 1).view(image.size(0), image.size(1), 1)
image = image.mul_(batch_stds).add_(batch_means)
image = image.sub_(torch.min(image)).div_(torch.max(image) - torch.min(image)).view(image.size(0), image.size(1), data.image_size, data.image_size)

ortho_image = ortho_image.view(ortho_image.size(0), ortho_image.size(1), -1)
batch_means = torch.tensor(data.mean).repeat(ortho_image.size(0), 1).view(ortho_image.size(0), ortho_image.size(1), 1)
batch_stds  = torch.tensor(data.std).repeat(ortho_image.size(0), 1).view(ortho_image.size(0), ortho_image.size(1), 1)
ortho_image = ortho_image.mul_(batch_stds).add_(batch_means)
ortho_image = ortho_image.sub_(torch.min(ortho_image)).div_(torch.max(ortho_image) - torch.min(ortho_image)).view(ortho_image.size(0), ortho_image.size(1), data.image_size, data.image_size)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2)

for i in list(range(2)):
    if i == 0:
        img = image
        ax1.set_xlabel("Original")
    else:
        img = ortho_image
        ax2.set_xlabel("Orthogonally Transformed")

    # Plot in cell
    img = tv.utils.make_grid(img)
    if i == 0:
        ax1.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))
        ax1.set_xticks([])
        ax1.set_yticks([])
    else:
        ax2.imshow(np.transpose(img.detach().cpu().numpy(), (1, 2, 0)))
        ax2.set_xticks([])
        ax2.set_yticks([])

         
fig.subplots_adjust(hspace = -0.4, wspace=0)
fig.suptitle(set_name, y = 0.82, fontsize = 18)
    
if save_only:
    plt.savefig('results/' + data.set_name + '/plots/OrthoImg.png')
else:
    plt.show()


