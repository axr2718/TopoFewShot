from dataclasses import dataclass
import torch
from torchvision.transforms import v2

@dataclass
class Config:
    """
    Configurations for the experiment.
    """
    # Seed
    seed: int = 42

    # Models
    swinv2: str = 'swinv2_small_window16_256.ms_in1k'
    vit: str = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
    vit_dino: str = 'vit_small_patch16_224.dino' 
    vit_dinov2: str = 'vit_small_patch14_dinov2.lvd142m'

    # Dataset paths
    dataset_path: str = '/mnt/d/data/miniImagenet/base.json'

    # Pretrain parameters
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    pretrain_epochs: int = 400

    # Cosine decay parameters
    max_lr = 2 * learning_rate
    min_lr = max_lr * .10
    warmup_steps = 20

    # Few-shot learning parameters.
    k_way: int = 5
    n_shot: int = 1
    n_query: int = 16
    
    num_episodes: int = 100

    image_transforms: v2 = v2.Compose([v2.Resize((224, 224)),
                                                 v2.ToDtype(torch.float32, scale=True),
                                                 v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    
    # Pre-training parameters.
    miniImagenet_transforms: v2 = v2.Compose([v2.Resize((256, 256)),
                                              v2.RandomHorizontalFlip(p=0.5),
                                              #v2.RandomVerticalFlip(p=0.5),
                                              v2.ColorJitter(brightness=0.4,
                                                             contrast=0.4,
                                                             saturation=0.4,
                                                             hue=0.1),
                                              v2.RandomGrayscale(p=0.1),
                                              v2.RandomRotation(degrees=(-30, 30)),
                                              v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), 
                                                                              sigma=(0.1, 5))], p=0.5),
                                              v2.RandAugment(num_ops=2, magnitude=9),
                                              v2.ToImage(),
                                              v2.ToDtype(torch.float32, scale=True),
                                              v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
