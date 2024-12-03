import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch

def train(model: nn.Module,
          support_dataset: Dataset,
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

    trainloader = DataLoader(dataset=support_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4)
    
    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item() * images.size(0)
        
        epoch_loss = total_loss / len(support_dataset)

        print(f'Epoch {epoch + 1}, Loss {epoch_loss:.5f}')

    return model