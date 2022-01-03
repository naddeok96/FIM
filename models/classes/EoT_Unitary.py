from art.preprocessing.expectation_over_transformation.pytorch import EoTPyTorch
from models.classes.first_layer_unitary_net import  FstLayUniNet
from typing import Optional, Tuple, TYPE_CHECKING, List
import warnings

class UniEoT(EoTPyTorch):

    def __init__(self,
                    data,
                    model_name,
                    nb_samples: int,
                    clip_values: Tuple[float, float],
                    apply_fit: bool = False,
                    apply_predict: bool = True,
                    gpu: bool = False,
                ) -> None:
        """
        Create an instance of EoTPyTorch.
        :param nb_samples: Number of random samples per input sample.
        :param clip_values: Tuple of float representing minimum and maximum values of input `(min, max)`.
        :param apply_fit: True if applied during fitting/training.
        :param apply_predict: True if applied during predicting.
        """
        super().__init__(nb_samples=nb_samples, clip_values=clip_values, apply_fit=apply_fit, apply_predict=apply_predict)

        self.data = data
        self.model_name = model_name
        self.gpu  = gpu
        self._device = "gpu" if self.gpu else "cpu"
            
        self.nb_samples = nb_samples
        self.clip_values = clip_values
        EoTPyTorch._check_params(self)

    def _transform(
        self, x: "torch.Tensor", y: Optional["torch.Tensor"], **kwargs
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"]]:
        """
        Internal method implementing the transformation per input sample.
        :param x: Input samples.
        :param y: Label of the samples `x`.
        :return: Transformed samples and labels.
        """
        # Get dimensions
        batch_size = x.size(0)
        channels   = x.size(1)
        height     = x.size(2)
        width      = x.size(3)

        # Load temporary network 
        ortho_net = FstLayUniNet(set_name = self.data.set_name,
                                gpu = self.gpu, 
<<<<<<< HEAD
                                model_name = 'cifar10_mobilenetv2_x1_0' if self.data.set_name == "CIFAR10" else "lenet")
=======
                                model_name = self.model_name)
>>>>>>> 35b62168acdf7bea07dd3aa8e4edf5b5b7026825

        # Set a random orthogonal matrix
        ortho_net.set_orthogonal_matrix()

        # Collect Stats for normalizing from entire dataset
        images, labels = next(iter(self.data.get_train_loader(int(30), num_workers = 0)))
        if self.gpu:
            images, labels = images.cuda(), labels.cuda()

        # Rotate Images
        train_ortho_images = ortho_net.orthogonal_operation(images)

        # Resize
        train_ortho_images = train_ortho_images.view(train_ortho_images.size(0), train_ortho_images.size(1), -1)
        
        # Calulate Stats
        means = train_ortho_images.mean(2).mean(0)
        stds  = train_ortho_images.std(2).mean(0)

        # Perform orthogonal transformation to given images
        x = x.view(batch_size, channels, height, width)
        ortho_images = ortho_net.orthogonal_operation(x)

        # Normalize
        ortho_images = ortho_images.view(ortho_images.size(0), ortho_images.size(1), -1)
        batch_means = means.repeat(ortho_images.size(0), 1).view(ortho_images.size(0), ortho_images.size(1), 1)
        batch_stds  = stds.repeat(ortho_images.size(0), 1).view(ortho_images.size(0), ortho_images.size(1), 1)
        ortho_images = ortho_images.sub_(batch_means).div_(batch_stds).view(batch_size, channels, height, width)

        return ortho_images, y