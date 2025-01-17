import torch.nn as nn
import torch
import timm
from datasets.miniImagenetBase import MiniImageNetBaseDataset
from utils.set_seed import set_seed
from utils.create_optimizer import create_optimizer
from config import Config
from experiment.pretrain import pretrain

torch.set_float32_matmul_precision('high')

if __name__ == '__main__':
    config = Config()
    seed = config.seed
    set_seed(seed)

    transform = config.miniImagenet_transforms
    dataset = MiniImageNetBaseDataset(json_path=config.dataset_path, transform=transform)

    if (torch.cuda.is_available()):
        print("Using GPU")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')

    num_classes = len(dataset.classes)
    model = timm.create_model(config.swinv2, num_classes=num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = create_optimizer(model=model,
                                 learning_rate=config.learning_rate, 
                                 weight_decay=config.weight_decay)
    
    model = torch.compile(model)

    trained_model = pretrain(model=model,
                             pretrain_dataset=dataset,
                             criterion=criterion,
                             optimizer=optimizer,
                             epochs=config.pretrain_epochs,
                             batch_size=config.batch_size,
                             device=device)
    
    print("Saving model...")
    torch.save(trained_model.state_dict(), './saved_models/vit_2_miniImageNet_base.pth')

