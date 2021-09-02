import torch
from torch import nn


class Dice:

    def __init__(self):
        pass

    def __call__(self, output: torch.Tensor, target: torch.Tensor):
        output = output.flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        inter = torch.sum(output * target, dim=1)
        union = torch.sum(output, dim=1) + torch.sum(target, dim=1)
        return (2 * inter / union.clamp(1e-6)).mean(dim=0)
    

class DiceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.dice = Dice()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        dice = self.dice(output, target)
        return 1 - dice