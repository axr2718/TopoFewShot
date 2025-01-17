import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset

def create_episode(dataset: Dataset, k_way: int, n_shot: int, n_query: int) -> tuple[Dataset, Dataset]:
    """
    Create a few-shot episode from the dataset.
    
    Args:
        dataset: ChestXrayDataset instance
        k_way: Number of classes
        n_shot: Number of samples per class in support set
        n_query: Number of samples per class in query set
        
    Returns:
        support_images: Tensor of support set images
        support_labels: Tensor of support set labels
        query_images: Tensor of query set images
        query_labels: Tensor of query set labels
    """

    all_labels = dataset.target_labels
    
    episode_classes = np.random.choice(all_labels, size=k_way, replace=False)
    
    
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    for class_idx, class_name in enumerate(episode_classes):
        class_mask = dataset.filtered_data['Finding Labels'] == class_name
        class_indices = np.where(class_mask)[0]
        
        selected_indices = np.random.choice(class_indices, size=n_shot + n_query, replace=False)
        
        support_indices = selected_indices[:n_shot]
        query_indices = selected_indices[n_shot:]
        
        for idx in support_indices:
            img, _ = dataset[idx]
            support_images.append(img)
            support_labels.append(class_idx)
            
        for idx in query_indices:
            img, _ = dataset[idx]
            query_images.append(img)
            query_labels.append(class_idx)
    
    support_images = torch.stack(support_images)
    support_labels = torch.tensor(support_labels)
    query_images = torch.stack(query_images)
    query_labels = torch.tensor(query_labels)

    support_dataset = TensorDataset(support_images, support_labels)
    query_dataset = TensorDataset(query_images, query_labels)
    
    return support_dataset, query_dataset