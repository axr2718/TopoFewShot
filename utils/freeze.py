import torch.nn as nn

def freeze(model: nn.Module):
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

        for param in model.head.parameters():
            param.requires_grad = True

    return model