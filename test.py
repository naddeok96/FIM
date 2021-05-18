from data_setup import Data
import matplotlib.pyplot as plt
import numpy as np
from models.classes.first_layer_unitary_resnet    import FstLayUniResNet
import torchvision

data = Data(gpu = False, set_name = "CIFAR10", data_augment = False)
net = FstLayUniResNet(gpu = False, set_name = "CIFAR10",
                       model_name = 'cifar10_mobilenetv2_x1_0',
                       pretrained = False)
net.set_orthogonal_matrix()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(data.get_train_loader(4))
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))