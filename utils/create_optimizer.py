import torch

def create_optimizer(model: torch.nn.Module,
                     learning_rate: float, 
                     weight_decay: float):
        param_dict = {pn: p for pn, p in model.named_parameters()}
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