import torch.nn as nn
import timm

class SwinV2(nn.Module):
    """Pre-trained SwinV2 Base Window 16 256 model with specified number of classes."""
    def __init__(self, num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
        """
        Args:
            num_classes (int): The number of classes of the output layer.
        """
        super().__init__()

        self.swin = timm.create_model('swinv2_base_window16_256.ms_in1k', pretrained=pretrained, num_classes=num_classes)

        if freeze:
            for param in self.swin.parameters():
                param.requires_grad = False

            for param in self.swin.head.fc.parameters():
                param.requires_grad = True
        

    def forward(self, x):
        """Run inference on the model."""
        return self.swin(x)