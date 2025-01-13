import torch.nn as nn
import torch
from torch.utils.data import Dataset
import copy
from .episode import create_episode
from .train import train
from .test import test
from utils.mean_std import compute_mean_std_err

def run_episodes(num_episodes: int,
                 dataset: Dataset, 
                 k_way: int, 
                 n_shot: int, 
                 n_query: int, 
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 epochs: int,
                 batch_size: int,
                 device: torch.device) -> tuple[float, float]:
    """
    Run a number of episodes on the model and dataset and get the average and standard error at the end.

    Args:
        num_episodes (int): The number of episodes to run the model and dataset on.
        dataset (Dataset): The entire dataset for which the experiment is ran on.
        k_way (int): The number of classes to be used for classification.
        n_shot (int): The number of support samples.
        n_query (int): The number of samples to be tested on.
        model (nn.Module): The model that is being trained and tested on.
        criterion (nn.Module): The loss function during training.
        optimizer (Optimizer): The optimizer during training.
        epochs (int): The number of epochs to train on.
        batch_size (int): The batch size for the DataLoader.
        device (device): The device the experiment is ran on.

    Returns:
        (float, float): The average of all the episodes and their corresponding standard error.
    """
    
    accuracies = []

    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}")
        support_dataset, query_dataset = create_episode(dataset=dataset,
                                                        k_way=k_way,
                                                        n_shot=n_shot,
                                                        n_query=n_query)
        
        trained_model = train(model=model,
                            support_dataset=support_dataset,
                            criterion=criterion,
                            optimizer=optimizer,
                            epochs=epochs,
                            batch_size=batch_size,
                            device=device)
        
        accuracy = test(model=trained_model,
                        query_dataset=query_dataset,
                        batch_size=batch_size, 
                        device=device)
    
        accuracies.append(accuracy)

        model.load_state_dict(initial_model_state)
        optimizer.load_state_dict(initial_optimizer_state)

    accuracies_mean, accuracies_std = compute_mean_std_err(accuracies=accuracies)
    print(f"Average accuracy across all episodes: {accuracies_mean} +/- {accuracies_std}")
    return accuracies_mean, accuracies_std