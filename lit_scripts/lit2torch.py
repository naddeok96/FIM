# Imports
from models.classes.lit_lenet import LitLeNet

# Initalize Unitary Model
Unet = LitLeNet.load_from_checkpoint(set_name=set_name,
                                    checkpoint_path = 'models/pretrained/Lit_LeNet_MNIST_mini_standard_U-val_acc=0.97.ckpt')
Unet.load_orthogonal_matrix("models/pretrained/LeNet_MNIST_mini_standard_U.pkl")

