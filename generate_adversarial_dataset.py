# Imports 
import sys
sys.path.insert(1, '../adversarial-robustness-toolbox/')
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import CarliniL2Method, ProjectedGradientDescent

import os
import torch
from data_setup import Data
from adversarial_attacks import Attacker
from models.classes.first_layer_unitary_net import FstLayUniNet

# Hypers
set_name   = 'MNIST'
start_image_index = 0
gpu        = True
gpu_number = "1"
batch_size = 1
from_ddp   = True
pretrained_weights_filename = "models/pretrained/MNIST/lenet_w_acc_97.pt"
net_name = pretrained_weights_filename.split(".")[-2].split("/")[-1]
attack_types = ["PGD"]
save_root = "../../../data/naddeok/mnist_adversarial_perturbations/"

# Ensure save folder exists
for attack in attack_types:
    if not os.path.isdir(save_root):
        os.mkdir(save_root)

    if not os.path.isdir(save_root + net_name + "/"):
        os.mkdir(save_root + net_name + "/")

    if not os.path.isdir(save_root + net_name + "/" + attack):
        os.mkdir(save_root + net_name + "/" + attack + "/")

if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Load Data
data = Data(set_name = set_name,
            test_batch_size = batch_size,
            gpu = gpu,
            maxmin=True)

# Load Source Network
state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

if from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    
net = FstLayUniNet( set_name   = set_name,
                    model_name = "lenet",
                    gpu = gpu)

attacker = Attacker(net = net, data = data, gpu = gpu)

criterion       = torch.nn.CrossEntropyLoss()
indv_criterion  = torch.nn.CrossEntropyLoss(reduction = 'none')

# Load PGD
if isinstance(gpu, bool):
    if gpu:
        device_num  = "cuda:0" 
        device_type = "gpu"
    else:
        device_num  = "cpu"
        device_type = "cpu"
    
else:
    device_num  = "cuda:" + str(gpu)
    device_type = "gpu"

classifier = PyTorchClassifier( model       = net,
                                nb_classes  = data.num_classes,
                                loss        = criterion,
                                clip_values = (float(data.test_pixel_min), float(data.test_pixel_max)),
                                input_shape = (data.num_channels, data.image_size, data.image_size),
                                device_type = device_type,
                                device_num  = device_num)

# Hyperparameters from "Towards Deep Learning Models Resistant to Adversarial Attacks"
if data.set_name == "MNIST":
    norm     = "inf"
    eps      = 0.3
    max_iter = 40
    eps_step = 0.01

elif data.set_name == "CIFAR10":
    norm     = "inf"
    eps      = 0.3 * 1.6
    max_iter = 40
    eps_step = 0.01 * 1.6

else:
    print("Eneter a valid data_set name for PGD")
    exit()
attack_pgd = ProjectedGradientDescent(  estimator  = classifier,
                                        norm       = norm,
                                        eps        = eps,     
                                        max_iter   = max_iter,
                                        eps_step   = eps_step, 
                                        batch_size = data.test_batch_size,
                                        targeted   = True, 
                                        verbose    = False)      
# Initialize CW2
# Hyperparameters from "Towards Deep Learning Models Resistant to Adversarial Attacks"
if data.set_name == "MNIST":
    max_iter = 25
elif data.set_name == "CIFAR10":
    max_iter = 20
else:
    print("Eneter a valid data_set name for PGD")
    exit()

attack_cw2 = CarliniL2Method(   classifier  = classifier,
                                max_iter    = max_iter, 
                                batch_size  = data.test_batch_size, 
                                verbose     = False)
                            
print("Test Set")
for attack in attack_types:
    print("Working on:", attack)
    for i, (images, labels) in enumerate(data.test_loader):
        print("Checking: ", i)
        # # Skip
        # if i < start_image_index: 
        #     print("Skip due to start index...")
        #     continue
        
        # # Give space
        # if count_down > 0:
        #     count_down -= 0
        #     print("Skip due to count down...")
        #     continue
        
        # # Check if done
        # already_done = False
        # for file in os.listdir("../../../data/naddeok/mnist_U_files/optimal_U_for_lenet_w_acc_98/test/"):
        
        #     if 'U{}'.format(i) == file:
        #         already_done = True
                
        #         count_down = 0
        #         break

        # if already_done:
        #     print("Skip due to already done...")
        #     continue
        
        # Display if made
        print("Working on: ", i)

        if gpu:
            images = images.cuda()
            labels = labels.cuda()

        # Generate normed attack
        if attack   == "OSSA":
            # Highest Eigenvalue and vector
            eig_vec_max, losses = attacker.get_max_eigenpair(images, labels)
            normed_attacks      = attacker.normalize(eig_vec_max, p = None, dim = 2)
        
        elif attack == "Gaussian_Noise":
            # Get losses
            outputs = net(images)
            losses  = indv_criterion(outputs, labels)

            # Generate attack
            normed_attacks = attacker.normalize(torch.rand_like(images.view(images.size(0), 1, -1)), p = None, dim = 2)

        elif attack == "FGSM":
            # Calculate Gradients
            gradients, batch_size, losses, predicted = attacker.get_gradients(images, labels)
            normed_attacks = attacker.normalize(torch.sign(gradients), p = None, dim = 2)

        elif attack == "PGD":
            # Get random targets
            import random
            targets = torch.empty_like(labels)
            for j, label in enumerate(labels):
                possible_targets = list(range(10))
                del possible_targets[label]

                targets[j] = random.choice(possible_targets)

            # Generate adversarial examples
            attacks = attack_pgd.generate(x=images.detach().cpu().numpy(), 
                                        y=targets.detach().cpu().numpy())
            attacks = torch.from_numpy(attacks)

            if isinstance(gpu, bool):
                if gpu:
                    attacks = attacks.cuda()
            else:
                attacks = attacks.to(gpu)
            
            # Get losses
            outputs = net(images)
            losses  = indv_criterion(outputs, labels)
            
            # Reduce the attacks to only the perturbations
            attacks = attacks - images

            # Norm the attack
            normed_attacks = attacker.normalize(attacks.view(batch_size, 1, -1), p = None, dim = 2)
            
        elif attack == "CW2":
            # Generate adversarial examples
            attacks = attack_cw2.generate(x=images.detach().cpu().numpy())
            attacks = torch.from_numpy(attacks)

            if isinstance(gpu, bool):
                if gpu:
                    attacks = attacks.cuda()
            else:
                attacks = attacks.to(gpu)
            
            # Get losses
            outputs = net(images)
            losses  = indv_criterion(outputs, labels)
            
            # Reduce the attacks to only the perturbations
            attacks = attacks - images

            # Norm the attack
            normed_attacks = attacker.normalize(attacks.view(batch_size, 1, -1), p = None, dim = 2)

        else:
            print("Invalid Attack Type")
            exit()


        # Save
        print(save_root + net_name + "/" + attack + "/P{}.pt".format(i))
        torch.save(normed_attacks, save_root + net_name + "/" + attack + "/P{}.pt".format(i))

        
print("donezo")

# Join all images into one dataset
# unitary_images = torch.empty((  int(1e4),
#                                 1,
#                                 28, 
#                                 28))

# for i in range(original_images.size(0)):
#     UA = torch.load(unitary_root + 'UA{}.pt'.format(i))
#     unitary_images[i,:,:,:] = UA

# torch.save((unitary_images, labels), '../../../data/naddeok/mnist_U_files/optimal_UA_for_lenet_w_acc_98/test/testing.pt')