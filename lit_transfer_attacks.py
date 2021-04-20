# Imports
import os
import wandb
import pickle
import torch
import pytorch_lightning as pl
from unorm import UnNormalize
from pytorch_lightning.loggers import WandbLogger
from models.classes.lit_lenet import LitLeNet
from pytorch_lightning.callbacks import ModelCheckpoint


# Parameters
set_name           = "MNIST"
wandb_project_name = "LeNet " + set_name + " Lightning"
wandb_mode         = "disabled" # "online", "offline" or "disabled"
run_name           = "OSSA"
gpus               = "2, 3"

# Initalize Weights and Biases
wandb_logger = WandbLogger(name = run_name, 
                            project=wandb_project_name,
                            mode = wandb_mode)

# Initalize Models
attacker_net = LitLeNet.load_from_checkpoint(
    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_no_U_attacker-val_acc=0.99.ckpt',
    set_name=set_name,
    learning_rate = 0.01, 
    momentum = 0.9, 
    weight_decay = 0.001,
    batch_size=256)
    
net = LitLeNet.load_from_checkpoint(
    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_standard_U-val_acc=0.89.ckpt',
    set_name=set_name,
    learning_rate = 0.01, 
    momentum = 0.9, 
    weight_decay = 0.001,
    batch_size=256)
net.load_orthogonal_matrix("models/pretrained/LeNet_MNIST_standard_U.pkl")

# Initialize Training
trainer = pl.Trainer(logger=wandb_logger,
                        gpus=gpus,            
                        accelerator='ddp')

# Attack
attacker_net.set_attack_type(attack_type = "OSSA",
                                transfer_network = net,
                                epsilons = [1])
trainer.test(attacker_net)
