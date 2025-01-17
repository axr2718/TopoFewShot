import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils import clip_grad_norm_
import math
from torchvision.transforms import v2

def cosine_decay(iteration: int,
                 max_lr: float,
                 warmup_steps: int,
                 max_steps: int):
    
    min_lr = max_lr * .1
    
    if (iteration < warmup_steps):
        return (max_lr * (iteration + 1)) / warmup_steps
    
    if (iteration > max_steps):
        return min_lr
    
    decay_ratio = (iteration - warmup_steps) / (max_steps - warmup_steps)

    coefficient = (1.0 + math.cos(math.pi * decay_ratio)) / 2

    return min_lr + coefficient * (max_lr - min_lr)

def pretrain(model: nn.Module,
             pretrain_dataset: Dataset,
             criterion: nn.Module,
             optimizer: torch.optim.Optimizer,
             epochs: int,
             batch_size: int,
             device: torch.device) -> nn.Module:
    """
    Trains a model on the dataset given the loss function, optimizer, and number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        support_dataset (Dataset): The dataset being used for training.
        criterion (nn.Module): The loss function.
        optimizer (Optimizer): Optimizer for training.
        epochs (int): Number of epochs to be trained on.
        batch_size (int): The batch size for the training DataLoader
        device (device): Device that will train.

    Returns:
        nn.Module: The trained model.
    """

    model.train()

    trainloader = DataLoader(dataset=pretrain_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8,
                             pin_memory=True)
    
    num_classes = len(trainloader.dataset.classes)
    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    
    max_steps = 90000

    max_lr = 1e-3
    warmup_steps = 9000
    
    accumulation_steps = 8
    iteration = 0
    for epoch in range(epochs):
        total_loss = 0.0

        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images, labels = cutmix_or_mixup(images, labels)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()

            norm = clip_grad_norm_(model.parameters(), max_norm=5.0)

            lr = cosine_decay(iteration=iteration,
                              max_lr=max_lr,
                              warmup_steps=warmup_steps,
                              max_steps=max_steps)
    
            print(f'step: {i} | loss: {loss.item():.5f} | lr: {lr:.8f} | norm: {norm:.4f}')
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.step()

            total_loss += loss.item() * images.size(0)
            iteration += 1
        
        epoch_loss = total_loss / len(pretrain_dataset)

        print(f'Epoch {epoch + 1}, Loss {epoch_loss:.4f}, Norm {norm:.4f}')

    return model
