from torch.utils.data import Dataset
import json
import os
from PIL import Image

class MiniImageNetBaseDataset(Dataset):
    """miniImageNet Base Classes Dataset"""
    def __init__(self, json_path, transform=None):
        """
        Args:
            json_path (str): Path to the base.json file
            transform (callable, optional): Transformations to be applied
        """
        self.data = update_json_paths(json_path)
        self.transform = transform
        
        self.classes = self.data['label_names']
        
        self.image_paths = self.data['image_names']
        self.labels = self.data['image_labels']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
def update_json_paths(json_path, new_prefix='./data/miniImagenet/images/'):
    """Update the image paths in the JSON file to match local directory structure."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    new_image_names = []
    for img_path in data['image_names']:
        parts = img_path.split('/')
        class_name = parts[-2]
        img_name = parts[-1]
        new_path = os.path.join(new_prefix, class_name, img_name)
        new_image_names.append(new_path)
    
    data['image_names'] = new_image_names
    return data