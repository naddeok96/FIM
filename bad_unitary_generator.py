# Imports 
import torch
import numpy as np
import pickle
from data_setup import Data
from models.classes.first_layer_unitary_net    import FstLayUniNet

def get_rotation_matrix(angle, degrees = True):
    '''
    Generates a 2x2 rotation matrix of input angle in rad
    '''
    # Convert to radians if using degrees
    if degrees:
        angle = np.deg2rad(angle)

    # Generate rotation matrix
    return torch.Tensor([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])

def embed_rotation_matrix(rotation, unitary, start_coords = [0, 0]):

    sub_unitary = unitary[start_coords[0] : start_coords[0] + rotation.size(0), start_coords[1] : start_coords[1] + rotation.size(1)]
    rotated = torch.mm(rotation, sub_unitary)
    unitary[start_coords[0] : start_coords[0] + rotation.size(0), start_coords[1] : start_coords[1] + rotation.size(1)] = rotated
    
    return unitary

# Set name
set_name = "MNIST"

# Set size of unitary matrix
if set_name == "MNIST":
    unitary_size = 28
    batch_size = int(5e4)
elif set_name == "CIFAR10":
    unitary_size = 32
    batch_size = int(5e4)

# Load Network
net = FstLayUniNet(set_name = set_name,
                    model_name="cifar10_mobilenetv2_x1_0" if set_name == "CIFAR10" else "lenet")

# Load Data
data = Data(set_name        = set_name,
            test_batch_size = batch_size)

# Generate weak U
R = get_rotation_matrix(1, degrees = True)
I = torch.eye(unitary_size)

net.U = embed_rotation_matrix(R, I, [0, 0])

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
filename = "weak_U_w_means_"
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