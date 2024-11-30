from dataclasses import dataclass
from torchvision.transforms import transforms


@dataclass
class Config:
    seed: int = 42

    k_way: int = 5
    n_shot: int = 50
    n_query: int = 16
    
    epochs: int = 100
    batch_size: int = 64
    num_episodes: int = 600

    image_transforms: transforms = transforms.Compose([transforms.Resize((256, 256)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      std=[0.229, 0.224, 0.225])])
    
    miniImagenet_epochs = 400