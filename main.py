import torch
import torch.nn as nn
from models.vision_model import VisionModel
import torch.optim as optim
from datasets.chest_xray import ChestXrayDataset
from utils import set_seed, create_optimizer
from config import Config
from experiment.run_episodes import run_episodes

if __name__ == '__main__':
    config = Config()
    seed = config.seed
    set_seed.set_seed(seed)

    if (torch.cuda.is_available()):
        print("Using GPU")
        device = torch.device('cuda')
    else:
        print("Using CPU")
        device = torch.device('cpu')
    
    transform = config.image_transforms
    dataset = ChestXrayDataset(csv_path='./data/chestx/Data_Entry_2017.csv', img_dir='./data/chestx/images', transform=transform)

    k_way = config.k_way
    n_shot = config.n_shot
    n_query = config.n_query


    num_classes = k_way

    pretrained_dict = torch.load('./saved_models/swinv2_miniImageNet_base.pth', weights_only=True)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if "head" not in k}
    pretrained_dict = {k.replace("_orig_mod.", ""): v for k, v in pretrained_dict.items()}

    model = VisionModel(model_name=config.model_name,
                        num_classes=num_classes, 
                        pretrained=False, 
                        freeze=True)

    model_dict = model.state_dict()
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)
    model = model.to(device)
    model = torch.compile(model=model)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer.create_optimizer(model=model,learning_rate=1e-1, weight_decay=0.001)

    epochs = config.epochs
    batch_size = config.batch_size
    num_episodes = config.num_episodes

    accuracy_mean, accuracy_std = run_episodes(num_episodes=num_episodes,
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
