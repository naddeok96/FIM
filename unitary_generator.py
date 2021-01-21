# Imports
import torch
import pickle
import numpy as np

# Generates N number of unitary matricies


def get_N_orthogonal_matricies(N, k = 1, input_size = 784):
    # Initialize 
    cos_sim = torch.nn.CosineSimilarity()
    I = torch.eye(input_size).view(1, input_size * input_size)
    U = np.empty((k*N, 1, input_size * input_size))
    rotations = np.empty((k*N))

    # Generate U's and calculate rotation
    for i in range(k*N):
        U[i] = torch.nn.init.orthogonal_(torch.empty(input_size, input_size)).view(1, input_size * input_size)

        rotations[i] = cos_sim(torch.from_numpy(U[i]), I).item()

    # Sort by rotations
    sorted_rotations_index = np.argsort(rotations)
    U = U[sorted_rotations_index]
    rotations = rotations[sorted_rotations_index]
    
    # Get N Us spaced as equally as possible
    Rs = [rotations[np.searchsorted(rotations, x)] for x in np.linspace(min(rotations), max(rotations), N)]
    Us = [torch.from_numpy(U[np.searchsorted(rotations, x)]).view(input_size, input_size) for x in np.linspace(min(rotations), max(rotations), N)]
    
    # Save 
    with open("models/pretrained/" + str(N) + "_Rs" + '.pkl', 'wb') as output:
        pickle.dump(Rs, output, pickle.HIGHEST_PROTOCOL)

    with open("models/pretrained/" + str(N) + "_Us" + '.pkl', 'wb') as output:
        pickle.dump(Us, output, pickle.HIGHEST_PROTOCOL)

# Main
get_N_orthogonal_matricies( N = 10,
                            k = 10**3)



# N = 10
# with open("models/pretrained/" + str(N) + "_Us" + '.pkl', 'rb') as input:
#     U = pickle.load(input)

        
# print(U[0].size())