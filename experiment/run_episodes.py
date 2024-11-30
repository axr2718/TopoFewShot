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