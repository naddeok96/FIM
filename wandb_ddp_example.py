import os
import numpy as np
import torch
import argparse
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torchvision.transforms as transforms
from mnist_sweep_config import sweep_config
import wandb

def parse_args():
    """
    Parse arguments given to the script.
    Returns:
        The parsed argument object.
    """
    parser = argparse.ArgumentParser(
        description="Run distributed data-parallel training and log with wandb.")
    # Used for `distribution.launch`
    parser.add_argument(
        "--local_rank", type=int, default=-1, metavar="N", help="Local process rank."
    )
    
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch",
        default=32,
        type=int,
        metavar="N",
        help="number of data samples in one batch",
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="wandb entity",
    )
    parser.add_argument(
        "--project",
        type=str,
        help="wandb project",
    )
    args = parser.parse_args()
    return args

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

def train(args, run):
    """
    Train method for the model.
    Args:
        args: The parsed argument object
        run: If logging, the wandb run object, otherwise None
    """
    # set the device
    total_devices = torch.cuda.device_count()
    device = torch.device(args.local_rank % total_devices)

    # initialize PyTorch distributed using environment variables
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(device)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(42)

    # initialize model -- no changes from normal training
    model = ConvNet()
    # send your model to GPU
    model = model.to(device)

    batch_size = args.batch
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Wrap the model
    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[device], output_device=device
    )

    # watch gradients only for rank 0
    if is_master:
        run.watch(model)

    # Data loading code
    train_dataset = torchvision.datasets.FashionMNIST(
        root='../../../data/pytorch/', train=True, transform=transforms.ToTensor(), download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=train_sampler,
    )

    total_step = len(train_loader)
    for epoch in range(args.epochs):
        batch_loss = []
        for ii, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = float(loss)
            batch_loss.append(loss)

            if (ii + 1) % 100 == 0 and is_master:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, 10, ii + 1, total_step, loss
                    )
                )
            if do_log:
                run.log({"batch_loss": loss})

        if do_log:
            run.log({"epoch": epoch, "loss": np.mean(batch_loss)})


if __name__ == "__main__":
    
    # Hyperparameters
    gpu_ids      = "4, 5"
    project_name = "DDP MNIST"

    # Set GPUs to Use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids


    # wandb.init a run if logging, otherwise return None
    run = wandb.init(
                entity="naddeok5",
                project=project_name,
            )

    train(args, run)