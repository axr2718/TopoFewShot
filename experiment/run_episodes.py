import torch.nn as nn
import torch
from torch.utils.data import Dataset
import numpy as np
from .episode import create_episode
from .train import train
from .test import test

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
                 device: torch.device) -> int:
    
    accuracies = []
    
    for episode in range(num_episodes):
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

    return np.mean(accuracies)