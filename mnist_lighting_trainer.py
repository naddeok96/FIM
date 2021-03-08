# Imports
import pytorch_lightning as pl
from models.classes.lightning_lenet import LitLeNet
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

# Parameters
set_name = "MNIST"
wandb_project_name = 'LeNet MNIST Lightning'
gpus = "0, 1"
n_epochs = 1


# Initalize Weights and Biases
wandb_logger = WandbLogger(project=wandb_project_name)


# Initalize Model
net = LitLeNet(set_name=set_name,
                batch_size=512)
# with open("models/pretrained/high_R_U.pkl", 'rb') as input:
#     net.U = pickle.load(input).type(torch.FloatTensor)
# print(net.U)
# exit()

checkpoint_callback = ModelCheckpoint(
    dirpath='/home/naddeok5/FIM/models/pretrained/',
    filename='Lit_LeNet_MNIST_{epoch:02d}-{val_acc:.2f}',
    save_top_k = 1,
    verbose=False,
    monitor='train_loss',
    mode='min',
    period = max(1, n_epochs/100))

trainer = pl.Trainer(logger=wandb_logger,
                        gpus=gpus,             
                        max_epochs=n_epochs,
                        accelerator='ddp',
                        checkpoint_callback=checkpoint_callback)

trainer.fit(net)
trainer.test()