import torch
from torch import nn
from pytorch_lightning import LightningModule
from collections import OrderedDict

class BasicBlock(nn.Module):

    def __init__(self, in_plane: int, plane: int, down_sample = False):
        super().__init__()

        if down_sample:
            self.skip_connection = nn.Conv2d(
                in_plane,
                plane,
                kernel_size=(2, 2),
                stride=2,
                padding=0
            )
        else:
            if in_plane == plane:
                self.skip_connection = None
            else:
                self.skip_connection = nn.Conv2d(
                    in_plane,
                    plane,
                    kernel_size=(3, 3),
                    stride=1,
                    padding=1
                )

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_plane, plane, kernel_size=3, stride=2 if down_sample else 1, padding=1, bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU(True)
        )

        self.conv2 = nn.Conv2d(plane, plane, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(plane)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.skip_connection is not None:
            out = self.skip_connection(x)
        else:
            out = x

        x = self.layer1(x)
        x = self.conv2(x)

        x += out
        x = self.bn(x)
        x = self.relu(x)
        return x

class Net(nn.Module):
    
    def __init__(self, in_channel = 3, ):
        super(Net, self).__init__()

        back_bone = OrderedDict()

        planes = 16

        back_bone["conv_1"] = nn.Sequential(
            BasicBlock(in_channel, planes, down_sample=True),
            *[BasicBlock(planes, planes) for _ in range(4) ]
        )
        planes *= 4

        back_bone["conv_2"] = nn.Sequential(
            BasicBlock(planes, planes*2, down_sample=True),
            *[BasicBlock(planes*2, planes*2) for _ in range(4)]
        )

        planes *= 4
        back_bone["conv_3"] = nn.Sequential(
            BasicBlock(planes, planes*2, down_sample=True),
            *[BasicBlock(planes*2, planes*2) for _ in range(2)]
        )

        planes *= 4
        back_bone["conv_4"] = nn.Sequential(
            BasicBlock(planes, planes*2, down_sample=True),
            *[BasicBlock(planes*2, planes*2) for _ in range(2)]
        )

        self.back_bone = nn.Sequential(back_bone)

        planes *= 4
        self.classify = nn.Sequential(
            nn.AdaptiveMaxPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(planes * 16, 128),

        )

    def forward(self, x):
        x = self.back_bone(x)
        return x


class LightningNet(LightningModule):

    def __init__(self, in_channel = 3):
        super(LightningNet, self).__init__()

        self.net = Net(in_channel)

    def forward(self, x):
        x = self

    def configure_optimizers(self):
        pass

    def train_step(self):
        pass

    def test_ste(self):
        pass