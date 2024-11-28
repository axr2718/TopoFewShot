import torch
import torch.nn as nn
from torchvision import transforms
from models.swinv2 import SwinV2
import torch.optim as optim
from datasets.chest_xray import ChestXrayDataset
from experiment.episode import create_episode
from utils.set_seed import set_seed
from config import Config
from experiment.run_episodes import run_episodes

if __name__ == '__main__':
    config = Config()
    seed = config.seed
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = config.chestxray_transforms
    dataset = ChestXrayDataset(csv_path='./data/chestx/Data_Entry_2017.csv', img_dir='./data/chestx/images', transform=transform)

    k_way = config.k_way
    n_shot = config.n_shot
    n_query = config.n_query


    num_classes = k_way
    model = SwinV2(num_classes=num_classes)
    model.to(device=device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters())

    epochs = config.epochs
    batch_size = config.batch_size
    num_episodes = config.num_episodes

    average_accuray = run_episodes(num_episodes=num_episodes,
                                   dataset=dataset,
                                   k_way=k_way,
                                   n_shot=n_shot,
                                   n_query=n_query,
                                   model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   device=device)
