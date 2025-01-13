from dataclasses import dataclass
#from torchvision.transforms import transforms
#from torchvision.transforms import RandAugment

import torchvision.transforms.v2 as transforms
from torchvision.transforms.v2 import RandAugment


@dataclass
class Config:
    """
    Configurations for the experiment.
    """
    # Seed
    seed: int = 42

    # Model args
    #model_name: str = 'swinv2_small_window16_256.ms_in1k'
    model_name: str = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
    dataset_path: str = './data/miniImagenet/base.json'
    pretrained: bool = False
    freeze: bool = False

    # Train and test parameters.
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    epochs: int = 500

    # Cosine decay parameters
    max_lr = 2 * learning_rate
    min_lr = max_lr * .10
    warmup_steps = 20

    # Few-shot learning parameters.
    k_way: int = 5
    n_shot: int = 5
    n_query: int = 16
    
    num_episodes: int = 100

    image_transforms: transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
    
    # Pre-training parameters.
    miniImagenet_transforms: transforms = transforms.Compose([transforms.Resize((224, 224)),
                                                              transforms.RandomHorizontalFlip(p=0.5),
                                                              #transforms.RandomVerticalFlip(p=0.5),
                                                              transforms.ColorJitter(brightness=0.4,
                                                                                     contrast=0.4,
                                                                                     saturation=0.4,
                                                                                     hue=0.1),
                                                              transforms.RandomGrayscale(p=0.1),
                                                              transforms.RandomRotation(degrees=(-30, 30)),
                                                              transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), 
                                                                                                              sigma=(0.1, 5))], p=0.5),
                                                              RandAugment(num_ops=2, magnitude=9),
                                                              transforms.ToTensor(),
                                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])
                                                              ])
