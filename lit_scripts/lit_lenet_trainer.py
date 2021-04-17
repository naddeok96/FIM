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
wandb_mode         = "online" # "online", "offline" or "disabled"
save_k_models      = 1
run_name           = "mini_standard_U"
gpus               = "4, 6"
n_epochs           = 100

# Initalize Weights and Biases
wandb_logger = WandbLogger(name = run_name, 
                            project=wandb_project_name,
                            mode = wandb_mode)

# Initalize Model
net = LitLeNet(set_name=set_name,
                learning_rate = 0.01, 
                momentum = 0.9, 
                weight_decay = 0.001,
                batch_size=256)
net, 
                       data,

# Setup Orthoganal Matrix
net.load_orthogonal_matrix("models/pretrained/LeNet_MNIST_mini_standard_U.pkl") # Load a pretrained U
# net.set_orthogonal_matrix() # Generate a new random U
# net.set_random_matrix() # Generate a new random matrix that is not unitary just random
# net.save_orthogonal_matrix("models/pretrained/LeNet_MNIST_" + run_name + ".pkl") # Save new U

# Setup Automatic Saving
checkpoint_callback = ModelCheckpoint(
    dirpath='/home/naddeok5/FIM/models/pretrained/',
    filename='Lit_LeNet_' + set_name + '_' + run_name + '-{val_acc:.2f}',
    save_top_k = save_k_models,
    verbose=False,
    monitor='train_loss',
    mode='min',
    period = max(1, n_epochs/100))

# Initialize Training
trainer = pl.Trainer(logger=wandb_logger,
                        gpus=gpus,             
                        max_epochs=n_epochs,
                        accelerator='ddp',
                        checkpoint_callback=checkpoint_callback)

# Train!
trainer.fit(net)
trainer.test(net)
