'''
This script will train a model and save it
'''
# Imports
import torch
import pickle
from data_setup import Data
from academy import Academy
import torch.nn.functional as F
from torchsummary import summary
from adversarial_attacks import Attacker
from models.classes.first_layer_unitary_net  import FstLayUniNet


# Hyperparameters
#------------------------------#
# Machine parameters
seed       = 3
gpu        = True
gpu_number = "2"

# Training parameters
n_epochs          = 5
batch_size        = 124
learning_rate     = 0.25
momentum          = 0.9
weight_decay      = 0.0001
distill           = False
distillation_temp = 20
use_SAM           = False

# Model parameters
model_name                  = "lenet" # "cifar10_mobilenetv2_x1_0"
U_filename                  = "models/pretrained/MNIST/weak_U_w_means_0-005893729627132416_and_stds_0-9862208962440491_.pt" # "models/pretrained/CIFAR10/weak_U_w_means_0-013919433578848839_0-014711719006299973_0-01946287229657173_and_stds_0-9984912872314453_0-9985936284065247_1-0006623268127441_.pt"
pretrained_weights_filename = None # "models/pretrained/MNIST/lenet_w_acc_98.pt"
pretrained_accuracy         = 100
from_ddp                    = False
save_model                  = True

# Data parameters
set_name  = "CIFAR10"
data_root = '../data/' + set_name

# Adversarial Training parameters
attack_type = None
epsilon     = 0.15
#------------------------------#

# Display
print("Epochs: ", n_epochs)
print("Pretrained: ", pretrained_weights_filename)
print("SAM: ", use_SAM)
print("Adversarial Training Type: ", attack_type)
print("U" , U_filename)
if attack_type is not None:
    print("Epsilon: ", epsilon)

# Push to GPU if necessary
if gpu:
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# Declare seed and initalize network
torch.manual_seed(seed)

# Load data
data = Data(gpu = gpu, 
            set_name = set_name,
            root = data_root)
print(set_name, "Data Loaded")

# Load Network
if pretrained_weights_filename:
    state_dict = torch.load(pretrained_weights_filename, map_location=torch.device('cpu'))

if from_ddp:  # Remove prefixes if from DDP
    torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")

net = FstLayUniNet( set_name = set_name,
                    gpu = gpu,
                    U_filename = U_filename,
                    model_name = model_name)

if distill: # Load teacher net
    teacher_net = FstLayUniNet( set_name = set_name,
                                gpu = gpu,
                                U_filename = None,
                                model_name = model_name)
    teacher_net.load_state_dict(state_dict)
    teacher_net.eval()

    summary(teacher_net, (data.num_channels, data.image_size, data.image_size))

else:
    teacher_net = None
    if pretrained_weights_filename:
        net.load_state_dict(state_dict)

net.eval()
summary(net, (data.num_channels, data.image_size, data.image_size))

print("Network Loaded")

# Training procedure
def train(net, data, gpu, n_epochs, batch_size, use_SAM, attack_type, epsilon, teacher_net):
    #Get training data
    train_loader = data.get_train_loader(batch_size)

    #Create optimizer and loss functions
    optimizer = torch.optim.SGD(net.parameters(), 
                                lr = learning_rate, 
                                momentum = momentum,
                                weight_decay = weight_decay)

    if use_SAM:
        optimizer = SAM(net.parameters(), optimizer)

    criterion = torch.nn.CrossEntropyLoss()

    #Loop for n_epochs
    for epoch in range(n_epochs):      
        for inputs, labels in train_loader:
            # Push to gpu
            if gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            # Adversarial Training
            if attack_type:
                # Initalize Attacker with current model parameters
                attacker = Attacker(net = net, data = data, gpu = gpu)

                # Generate attacks and replace them with orginal images
                inputs = attacker.get_attack_accuracy(attack = attack_type,
                                                        attack_images = inputs,
                                                        attack_labels = labels,
                                                        epsilons = [epsilon],
                                                        return_attacks_only = True,
                                                        prog_bar = False)

            #Set the parameter gradients to zero
            optimizer.zero_grad()   

            #Forward pass
            outputs = net(inputs)              # Forward pass

            # Get loss
            if distill:
                # Proof that this method is equivalent to oringal criterion
                # batch_size = labels.size(0)
                # label_onehot = torch.FloatTensor(batch_size, data.num_classes)
                # label_onehot.zero_()
                # label_onehot.scatter_(1, labels.view(-1, 1), 1)
                # print("One Hot", label_onehot[0])
                # print(torch.sum(-label_onehot * F.log_softmax(outputs, -1), -1).mean())
                
                soft_labels = F.log_softmax(teacher_net(inputs), -1)
                loss = torch.sum(-soft_labels * F.log_softmax(outputs, -1), -1).mean()

            else:
                loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()                   # Find the gradient for each parameter
            optimizer.step()                  # Parameter update
        
        if epoch % 2 == 0:
            accuracy = test(net, data, gpu)
            print("Epoch: ", epoch + 1 , "\tLoss: ", loss.item(), "\tAcc:", accuracy)    

    return net 

# Evaluation procedure   
def test(net, data, gpu):
    # Initialize
    total_tested = 0
    correct = 0

    # Test images in test loader
    for inputs, labels in data.test_loader:
        # Push to gpu
        if gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        #Forward pass
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)

        # Update runnin sum
        total_tested += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate accuracy
    accuracy = (correct/total_tested)
    return accuracy

# Fit Model
print("Training")
net = train(net, data, gpu, n_epochs, batch_size, use_SAM, attack_type, epsilon, teacher_net)

# Calculate accuracy on test set
print("Testing")
accuracy = test(net, data, gpu)
print("Accuarcy: ", accuracy)

# Save Model
if save_model:
    # Define File Names
    filename  = "weak_U_" + model_name + "_w_acc_" + str(int(round(accuracy * 100, 3))) + ".pt"
    
    # Save Models
    torch.save(net.state_dict(), "models/pretrained/" + set_name + "/" + filename)

    # Save U
    if net.U is not None:
        torch.save(net.U, "models/pretrained/" + set_name + "/U_" + filename)
