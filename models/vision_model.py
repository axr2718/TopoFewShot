import torch.nn as nn
import timm
import torch

class VisionModel(nn.Module):
    """Create a vision model with specified number of classes."""
    def __init__(self, 
                 model_name: str,
                 num_classes: int, 
                 pretrained: bool = True, 
                 freeze: bool = True,
                 drop_rate=0.0,
                 drop_path_rate=0.3) -> nn.Module:
        """
        Args:
            num_classes (int): The number of classes of the output layer.
        """
        super().__init__()

        self.vision_model = timm.create_model(model_name=model_name, 
                                              pretrained=pretrained, 
                                              num_classes=num_classes,
                                              drop_rate=drop_rate,
                                              drop_path_rate=drop_path_rate)

        if freeze:
            for param in self.vision_model.parameters():
                param.requires_grad = False

            for param in self.vision_model.head.fc.parameters():
                param.requires_grad = True
        

    def forward(self, x):
        """Run inference on the model."""
        return self.vision_model(x)
    
    def create_optimizer(self, learning_rate: float, weight_decay: float):
        param_dict = {pn: p for pn, p in self.vision_transformer.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [{'params': decay_params, 'weight_decay': weight_decay},
                        {'params': nodecay_params, 'weight_decay': 0.0}]
        
        optimizer = torch.optim.AdamW(params=optim_groups,
                                      betas=(0.9, 0.95),
                                      lr=learning_rate,
                                      fused=True)
        
        return optimizer
