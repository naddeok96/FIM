# Imports
from sklearn.metrics import accuracy_score
import wandb
import os
import torch
from torch import normal
from data_setup import Data
from unitary_data_setup import UnitaryData
from models.classes.first_layer_unitary_net  import FstLayUniNet
import time

start   = time.time()

def unnormalize_image(x):
    return 255 * ((x  * data.std[0]) + data.mean[0])



def normalize_image(x):
    return (((x / 255) - data.mean[0]) / data.std[0])

def normalize_pert(x):
    return (((x / 255)) / data.std[0])


# Hypers
project_name                = "Optimal U Robustness Test"
set_name                    = 'MNIST'

# # Attacker
# pretrained_weights_filename = "models/pretrained/MNIST/MNIST_Models_for_Optimal_U_stellar-rain-5.pt"
# unitary_root    = None

# # Standard Black Box
# pretrained_weights_filename = "models/pretrained/MNIST/MNIST_Models_for_Optimal_U_lilac-butterfly-10.pt"
# unitary_root = None

# # # U Standard Black Box
# pretrained_weights_filename = "models/pretrained/MNIST/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5_MNIST_Models_for_Optimal_U_spring-smoke-13.pt"
# unitary_root = "../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test/"

# U Attacker Black Box
pretrained_weights_filename = "models/pretrained/MNIST/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5_MNIST_Models_for_Optimal_U_glad-dust-14.pt"
unitary_root    = "../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test/"

from_ddp                    = True
gpu                         = True

gpu_number                  = "3"

attack                      = "FGSM"
epsilons                    =  torch.linspace(0,255,52, dtype=torch.uint8).tolist()

pert_root       = "../../../data/naddeok/optimal_U_for_MNIST_Models_for_Optimal_U_stellar-rain-5/test/adversarial_perturbations/"

if project_name:
    run = wandb.init(
                    config  =   {
                        "pretrained_weights_filename" : pretrained_weights_filename.split('.')[0].split('/')[-1],
                        "attack_type" : attack,
                        "U" : "Optimal" if unitary_root is not None else None
                    },
                    entity  = "naddeok",
                    project = project_name)

# Declare GPUS
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Load Network
state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

if from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
    
net = FstLayUniNet( set_name   = set_name,
                    model_name = "lenet",
                    gpu = gpu)
net.load_state_dict(state_dict)
net.eval()

# Load Data
data = Data(set_name = set_name,
            test_batch_size = 1,
            gpu = gpu,
            maxmin=True)

# Declare criterion
criterion = torch.nn.CrossEntropyLoss(reduction='sum')
            
print("Setup time:", time.time() - start)
start   = time.time()

# Apply Perturbations
losses  = [_ for _ in range(len(epsilons))]
correct = [_ for _ in range(len(epsilons))]
for i, (image, label) in enumerate(data.test_loader):
    if (i+1)%1000 == 0:
        print("Image #: ", i+1)

    # Load Adversarial Perturbation
    pert = torch.load(pert_root + attack + "/P{}.pt".format(i))

    # Load Optimal U for image
    if unitary_root is not None:
        net.U = torch.load(unitary_root + 'U{}'.format(i))

    # Push to GPUs
    if gpu:
        image = image.cuda()
        label = label.cuda()
        pert  = pert.cuda()

    # Resize for easy addition
    image   = image.view(1,1,-1)
    pert    = pert.view(1,1,-1)
    assert set_name == "MNIST", "Rescale to 255 is only supported for single channel"
    
    # Get unnormalized image to add pert
    unnorm_image = unnormalize_image(image)

    # Cycle over all epsilons
    for j, epsilon in enumerate(epsilons):
        # Scale pert and add to image
        scaled_pert = epsilon * pert
        adv_image = unnorm_image + scaled_pert;
        adv_image   = normalize_image(torch.clamp(adv_image, 0, 255))

        with torch.no_grad(): 
            # Feedforward
            adv_outputs  = net(adv_image.view(1,1,28,28))

            adv_loss       = criterion(adv_outputs, label)
            _, adv_predictions  = torch.max(adv_outputs, 1)

            if epsilon == 0:
                loss = adv_loss.item()

            if epsilon !=0 and adv_loss.item() < loss and (label == adv_predictions):
                original_adv_predictions = adv_predictions
                adv_image   = normalize_image(torch.clamp(unnorm_image - scaled_pert, 0, 255))
                adv_outputs = net(adv_image.view(1,1,28,28))
                adv_loss    = criterion(adv_outputs, label)
                _, adv_predictions  = torch.max(adv_outputs, 1)

                if original_adv_predictions != adv_predictions:
                    print("Label", label)
                    print("Orginal Prediction:",original_adv_predictions)
                    print("Flipped Prediction:",adv_predictions)

            losses[j]   = losses[j]     + adv_loss.item()
            correct[j]  = correct[j]    + (label == adv_predictions).item()

            if (i + 1) == int(1e4):
                base_accuracy = correct[0] / (i + 1)
                accuracy = correct[j] / (i + 1)
                run.log({   "Epsilon"       : epsilon, 
                            "Loss"          : losses[j] / (i + 1),
                            "Accuracy"      : accuracy,
                            "Base Accuracy" : base_accuracy,
                            "Fooling Rate"  : (base_accuracy - accuracy) / base_accuracy 
                            })


print("Dataset Time:", time.time() - start)
wandb.finish()
print("Bye")
