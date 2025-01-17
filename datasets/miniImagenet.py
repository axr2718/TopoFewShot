import torch
import os
from PIL import Image
from torch.utils.data import Dataset

class MiniImageNetDataset(Dataset):
    """miniImageNet Dataset"""
    def __init__(self, dir_path, transform=None) -> Dataset:
        """
        Args:
            data_dir (str): Path to the root directory of miniImageNet.
            transform (callable, optional): Transformations to be applied to the images.
        """
        self.data_dir = dir_path
        self.transform = transform
        
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(dir_path))
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        
        for class_name in self.classes:
            class_folder = os.path.join(dir_path, class_name)
            for img_file in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_file)
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index of the image to retrieve.
        Returns:
            A tuple (image, label), where:
              - image (Tensor): The transformed image.
              - label (int): The class label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label