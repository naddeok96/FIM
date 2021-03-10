# Imports
import os
import wandb
import pickle
import torch
import namegenerator
import pytorch_lightning as pl
from unorm import UnNormalize
from pytorch_lightning.loggers import WandbLogger
from models.classes.lightning_lenet import LitLeNet
from pytorch_lightning.callbacks import ModelCheckpoint


# Parameters
set_name = "MNIST"
wandb_project_name = 'LeNet MNIST Lightning'
run_name = "random_U"
gpus = "0,1,2"
n_epochs = 1000


# Initalize Weights and Biases
wandb_logger = WandbLogger(name = run_name, project=wandb_project_name)

# Initalize Model
net = LitLeNet(set_name=set_name,
                batch_size=512)


# Setup Orthoganal Matrix
# net.load_orthogonal_matrix("models/pretrained/high_R_U.pkl")
# net.set_orthogonal_matrix()
net.set_random_matrix()
net.save_orthogonal_matrix("models/pretrained/LeNet_MNIST_" + run_name + ".pkl")

# Setup Automatic Saving
checkpoint_callback = ModelCheckpoint(
    dirpath='/home/naddeok5/FIM/models/pretrained/',
    filename='Lit_LeNet_MNIST_' + run_name + '-{val_acc:.2f}',
    save_top_k = 1,
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
trainer.test()
