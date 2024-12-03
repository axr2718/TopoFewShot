import torch.nn as nn
import torch

class BettiEncoder(nn.Module):
    def __init__(self, 
                 seq_length: int = 100,
                 d_model: int = 32,
                 nhead: int = 2,
                 num_layers: int = 1,
                 dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_embedding = nn.Linear(1, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation="relu",
                                                   batch_first=True)
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                                         num_layers=num_layers)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.input_embedding(x)
        x = x + self.pos_embedding
        x = self.transformer_encoder(x)

        return x
    
class BettiEncoderClassifier(nn.Module):
    def __init__(self, 
                 seq_length: int = 100,
                 d_model: int = 512,  
                 nhead: int = 4,      
                 num_layers: int = 4,  
                 dim_feedforward: int = 256,  
                 dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()
        
        self.encoder = BettiEncoder(seq_length=seq_length,
                                    d_model=d_model,
                                    nhead=nhead,
                                    num_layers=num_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):  
       
        x = self.encoder(x)
        
       
        x = torch.mean(x, dim=1) 
        
        x = self.norm(x)
        x = self.classifier(x)
        
        return x