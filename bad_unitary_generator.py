# Imports 
import torch
import numpy as np
import pickle

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

R = get_rotation_matrix(1, degrees = True)
I = torch.eye(28)

U = embed_rotation_matrix(R, I, [0, 0])
cos_sim = torch.nn.CosineSimilarity()
rotation_angle = cos_sim(U.view(1, -1), I.view(1, -1))

# # Save 
# with open("models/pretrained/high_R.pkl", 'wb') as output:
#     pickle.dump(rotation_angle, output, pickle.HIGHEST_PROTOCOL)

with open("models/pretrained//MNIST/weak_U" + '.pkl', 'wb') as output:
    pickle.dump(U, output, pickle.HIGHEST_PROTOCOL)

















# I = torch.eye(2)

# RI = torch.mm(R, I)

# cos_sim = torch.nn.CosineSimilarity()

# print(I, RI, type(I), type(RI))
# print(cos_sim(I.view(1, -1), RI.view(1, -1)).item())