import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

def test(model: nn.Module, 
         query_dataset: Dataset, 
         batch_size: int,
         device: torch.device) -> int:
    """
    Tests the performance of a model on a dataset.

    Args:
        model (nn.Module): The model to be tested on.
        query_dataset (Dataset): The dataset the model will use to test.
        batch_size (int): The batch size for the training DataLoader
        device (device): The device the model and dataset will be in.
        
    Returns:
        int: The accuracy of the test.
    """

    model.eval()

    testloader = DataLoader(dataset=query_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=8,
                            pin_memory=True)
    
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * (correct / total)

    print(f"Accuracy = {accuracy}")

    return accuracy

