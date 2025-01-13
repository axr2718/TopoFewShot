import torch.nn as nn
import torch

class BettiEncoder(nn.Module):
    def __init__(self, 
                 seq_length: int,
                 d_model: int,
                 nhead: int,
                 num_layers: int,
                 dim_feedforward: int,
                 dropout: float,
                 num_classes: int,
                 classifier: bool = False):
        super().__init__()
        self.classifier = classifier
        self.input_embedding = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation="relu",
                                                   batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=num_layers)
        
        self.classification_head = nn.Sequential(nn.Linear(d_model, num_classes))
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embedding(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)

        if self.classifier:
            x = torch.mean(x, dim=1)
            x = self.classification_head(x)

        return x
