import torch.nn as nn
import torch
from models.swinv2 import SwinV2
from datasets.miniImagenetBase import MiniImageNetBaseDataset
from utils.set_seed import set_seed
from config import Config
from experiment.train import train


if __name__ == '__main__':
    config = Config()
    seed = config.seed
    set_seed(seed)

    transform = config.image_transforms
    dataset = MiniImageNetBaseDataset(json_path='./data/miniImagenet/base.json', transform=transform)

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    num_classes = len(dataset.classes)
    model = SwinV2(num_classes=num_classes, pretrained=False, freeze=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.9)

    batch_size = config.batch_size
    epochs = config.miniImagenet_epochs
    
    trained_model = train(model=model,
                          support_dataset=dataset,
                          criterion=criterion,
                          optimizer=optimizer,
                          epochs=epochs,
                          batch_size=batch_size,
                          device=device)
    
    torch.save(model.state_dict(), './saved_models/swinv2_miniImageNet_base_best.pth')

