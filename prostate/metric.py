import torch
from torch import nn


class Dice:

    def __init__(self):
        pass

    def forward(self, output: torch.Tensor, target: torch.Tensor, smooth = 1):
        inter = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        return (2 * inter + smooth) / (union + smooth)
    

class DiceLoss(nn.Module):
    
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        smooth = 1
        inter = torch.sum(output * target)
        union = torch.sum(output) + torch.sum(target)
        return 1 - (2 * inter + 1) / (union + 1)