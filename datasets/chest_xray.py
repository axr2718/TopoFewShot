import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDataset(Dataset):
    """NIH Chest X-ray Dataset for Few-Shot Learning"""
    
    def __init__(self, csv_path: str, img_dir: str, transform=None) -> Dataset:
        """
        Args:
            csv_path (str): Path to the CSV file with annotations
            img_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.transform = transform
        self.data_frame = pd.read_csv(csv_path)
        
        self.target_labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumothorax"]
        
        self.label_to_idx = {label: idx for idx, label in enumerate(self.target_labels)}
        

        single_label_mask = ~self.data_frame['Finding Labels'].str.contains(r'\|')
        single_label_data = self.data_frame[single_label_mask].copy()
        target_mask = single_label_data['Finding Labels'].isin(self.target_labels)
        self.filtered_data = single_label_data[target_mask]
        
        self.img_dir = img_dir

    def __len__(self) -> int:
        """Return the number of images in the dataset"""
        return len(self.filtered_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Return an image and its label"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.filtered_data.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        label = self.filtered_data.iloc[idx]['Finding Labels']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.label_to_idx[label]
        
        return image, label_idx