# Imports
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from models.classes.lightning_lenet import LitLeNet

# Parameters
set_name = "MNIST"
wandb_project_name = 'LeNet MNIST Lightning'
gpus = "0, 1"

# Initalize Weights and Biases
wandb_logger = WandbLogger(project=wandb_project_name)

# Initalize Model
model = LitLeNet.load_from_checkpoint('models/pretrained/Lit_LeNet_MNIST_epoch=99-val_acc=0.969.ckpt', set_name=set_name)
trainer =  pl.Trainer(logger=wandb_logger,
                        gpus=gpus,
                        accelerator='ddp')
trainer.test(model)