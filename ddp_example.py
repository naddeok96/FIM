import os
import wandb
import sys
import tempfile
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.multiprocessing as mp
import torchvision
from models.classes.first_layer_unitary_net  import FstLayUniNet
from torch.nn.parallel import DistributedDataParallel as DDP
from mnist_sweep_config import sweep_config

def setup(rank, world_size, use_wandb):
    # Setup rendezvous
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initalize WandB logging on rank 0
    if rank == 0 and use_wandb:
        run = wandb.init(
                entity  = "naddeok",
                project = "DDP MNIST")
    else:
        run = None

    # Initialize the process group
    dist.init_process_group(backend    = "nccl",
                           init_method = "env://", 
                           rank        = rank, 
                           world_size  = world_size)
    return run

def cleanup():
    # Close all processes
    dist.destroy_process_group()

def train(rank, world_size, epochs, batch_size, use_wandb):
    # Initialize WandB and Process Group
    print(f"Training on rank {rank}.")
    run = setup(rank, world_size, use_wandb)

    # Create model and move it to GPU with id rank
    model = FstLayUniNet(   set_name = "MNIST", 
                            gpu = rank,
                            U_filename = None,
                            model_name = "lenet",
                            pretrained = False).to(rank)

    ddp_model = DDP(model, device_ids=[rank])

    # watch gradients only for rank 0
    if rank == 0 and run is not None:
        run.watch(model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='../../../data/pytorch/', 
                                                train=True, 
                                                transform=transforms.ToTensor(), 
                                                download=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader( dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=0,
                                                pin_memory=True,
                                                sampler=train_sampler)

    total_step = len(train_loader)
    for epoch in range(epochs):
        batch_loss = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(rank, non_blocking=True)
            labels = labels.to(rank, non_blocking=True)

            # Forward pass
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = float(loss)
            batch_loss.append(loss)

            # Display
            if (i + 1) % 2 == 0 and rank == 0:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, epochs, i + 1, total_step, loss))

                # Log
                if run is not None:
                    run.log({"batch_loss": loss})

        if run is not None and rank == 0:
            run.log({"epoch": epoch, "loss": np.mean(batch_loss)})

    
    # End of training
    print("Done Training on Rank ", rank)

    # Shut down gpu processes
    dist.barrier()
    cleanup()

    # Close WandB
    if rank == 0 and run is not None:
        wandb.finish()
        print("WandB Finished")

    
    

def run_ddp(func, world_size, epochs, batch_size, use_wandb):

    # Spawn processes on gpus
    mp.spawn(func,
             args=(world_size, epochs, batch_size, use_wandb,),
             nprocs=world_size,
             join=True)

    # Display
    print("Run Complete")

if __name__ == "__main__":
    # Hyperparameters
    gpu_ids      = "2, 6"
    project_name = "DDP MNIST"
    epochs       = 2
    batch_size   = int(2**9)
    use_wandb    = True

    # Set GPUs to Use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids

    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"

    # Run training using DDP
    run_ddp(train, n_gpus, epochs, batch_size, use_wandb)
