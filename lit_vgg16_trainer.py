# Imports
import os
import wandb
import pickle
import torch
import namegenerator
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from models.classes.lit_vgg16 import LitVGG16
from pytorch_lightning.callbacks import ModelCheckpoint


# Parameters
wandb_project_name = 'VGG16 ImageNet Lightning'
wandb_mode         = "offline" # "online", "offline" or "disabled"
save_k_models      = 1
run_name = "standard_U"
gpus = "2"
n_epochs = 0
batch_size = 1
pretrained = False

# Initalize Weights and Biases
wandb_logger = WandbLogger(name = run_name, 
                           project=wandb_project_name,
                           mode = wandb_mode)

# Initalize Model
net = LitVGG16(batch_size=batch_size,
                pretrained=pretrained)


# Setup Orthoganal Matrix
# net.load_orthogonal_matrix("models/pretrained/high_R_U.pkl")
net.set_orthogonal_matrix()
exit()
# net.set_random_matrix()
net.save_orthogonal_matrix("models/pretrained/LeNet_MNIST_" + run_name + ".pkl")

exit()

# Setup Automatic Saving
checkpoint_callback = ModelCheckpoint(
    dirpath='/home/naddeok5/FIM/models/pretrained/',
    filename='Lit_VGG16_ImageNet_' + run_name + '-{val_acc:.2f}',
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
