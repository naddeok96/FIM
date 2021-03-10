# Imports
import os
import wandb
import pickle
import torch
import namegenerator
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.classes.lightning_vgg16 import LitVGG16
from pytorch_lightning.callbacks import ModelCheckpoint


# Parameters
wandb_project_name = 'VGG16 ImageNet Lightning'
run_name = "standard_U"
gpus = "0,1,2,3"
n_epochs = 1000
batch_size = 1
pretrained = False


# Initalize Weights and Biases
wandb_logger = WandbLogger(name = run_name, project=wandb_project_name)

# Initalize Model
net = LitVGG16(batch_size=batch_size,
                pretrained=pretrained)


# Setup Orthoganal Matrix
# net.load_orthogonal_matrix("models/pretrained/high_R_U.pkl")
net.set_orthogonal_matrix()
# net.set_random_matrix()
net.save_orthogonal_matrix("models/pretrained/LeNet_MNIST_" + run_name + ".pkl")

# Setup Automatic Saving
checkpoint_callback = ModelCheckpoint(
    dirpath='/home/naddeok5/FIM/models/pretrained/',
    filename='Lit_VGG16_ImageNet_' + run_name + '-{val_acc:.2f}',
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
trainer.test(net)
