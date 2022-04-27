from data_setup import Data
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.classes.first_layer_unitary_net    import FstLayUniNet
from data_setup import Data
import torchvision.transforms as transforms

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

    for i in range(img.size(1)):
        
        print("Before: ", torch.min(img[:, i, :, :]), torch.max(img[:, i, :, :]))
        img[:, i, :, :] = (img[:, i, :, :] - torch.min(img[:, i, :, :])) / (torch.max(img[:, i, :, :]) - torch.min(img[:, i, :, :]))
        print("After: ", torch.min(img[:, i, :, :]), torch.max(img[:, i, :, :]))
    return img

# Hypers
set_name = 'MNIST'
batch_size = int(5e4)

# Load Data
data = Data(set_name        = set_name,
            test_batch_size = batch_size)

source_image, _ = next(iter(data.get_train_loader(batch_size=1)))

# Load Source Network
net = FstLayUniNet(set_name = set_name,
                    model_name="lenet",
                    pretrained_weights_filename="models\pretrained\MNIST\lenet_w_acc_98.pt")

# Set a random orthogonal matrix
net.set_orthogonal_matrix(source_image)

# Collect Stats
images, labels = next(iter(data.test_loader))

# Rotate Images
ortho_images = net.orthogonal_operation(images)

# Resize
ortho_images = ortho_images.view(ortho_images.size(0), ortho_images.size(1), -1)

# Calulate Stats
means = ortho_images.mean(2).mean(0)
stds  = ortho_images.std(2).mean(0)

# Generate Saving name with stats encoded
filename = "Optimal_U_w_means_"
for mean in means:
    strmean = str(mean.item())
    # print(strmean)
    for c in strmean:
        if c == "-":
            filename += "n"
        elif c == ".":
            filename += "-"
        else:
            filename += c
    filename += "_"

filename += "and_stds_"
for std in stds:
    strstd = str(std.item())
    for c in strstd:
        if c == "-":
            filename += "n"
        elif c == ".":
            filename += "-"
        else:
            filename += c
    filename += "_"
filename += ".pt"
print(filename)
torch.save(net.U, "models/pretrained/" + set_name + "/" + filename)


# # Decode filename
# stats = ["", "", "", "", "", ""]
# state = 0
# sofar = ""
# for c in filename:        
#     sofar += c
#     if sofar == "U_w_means_":
#         state = 1
#         continue
    
#     if state == 4 and "and_stds_" not in sofar:
#         continue

#     if state == 4 and len(stats[state - 1]) == 0 and c == "_":
#         continue

#     if 0 < state and state < 7:
#         if c == "n":
#             stats[state - 1] += "-"
#         elif c == "-":
#             stats[state - 1] += "."
#         elif c == "_":
#             stats[state - 1] = float(stats[state-1])
#             state += 1
#         else: 
#             stats[state - 1] += c

# means = torch.tensor(stats[0:3])
# stds  = torch.tensor(stats[3:])