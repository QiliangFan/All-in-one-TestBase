import torch
from torch import nn
from resnet import ResNet

class ClassifyNet(nn.Module):

    def __init__(self, num_class):
        super().__init__()

        self.backbone = ResNet(in_channel=1, layers=34)

        in_channel = self.backbone.last_channel
        pool_size = 16
        self.classifier = nn.ModuleDict({
            "adaptive_pool": nn.AdaptiveMaxPool2d((pool_size, pool_size)),
            "flattern": nn.Flatten(),
            "Linear": nn.Sequential(
                nn.Linear(in_channel * pool_size * pool_size, in_channel),
                nn.Linear(in_channel, in_channel),
                nn.Linear(in_channel, num_class)
            ),
            "Softmax": nn.Softmax(dim=1)
        })

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x