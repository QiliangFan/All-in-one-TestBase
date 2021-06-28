from torch import nn
from collections import OrderedDict
from torch.nn import Sequential


class BasicBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, down_sample = False):
        super().__init__()
        if down_sample:
            stride = 2
            self.short_cut = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        else:
            stride = 1
            self.short_cut = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=stride)
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        if self.short_cut is not None:
            identity = self.short_cut(x)
        else:
            identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x += identity
        return x

class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel: int, out_channel: int, down_sample = False):
        super().__init__()

        if down_sample:
            stride = 2
            self.short_cut = nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=3, padding=1, stride=stride)
        else:
            stride = 1
            self.short_cut = nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, padding=0, stride=1)
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion),
            nn.ReLU()
        )

    def forward(self, x):
        identity = self.short_cut(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += identity

        return x

class ResNet(nn.Module):
    def __init__(self, in_channel: int = 3, layers: int = 34):
        super().__init__()

        self.blocks = []
        if layers == 34:
            self.blocks = [3, 4, 6, 3]
            Block = BasicBlock
        elif layers == 50:
            self.blocks = [3, 4, 6, 3]
            Block = BottleneckBlock
        else:
            raise ValueError()
        assert len(self.blocks) > 0
        
        ch = 64
        self.layer1 = nn.Conv2d(in_channel, ch, kernel_size=7, padding=3, stride=2)

        layer2 = OrderedDict()
        layer2["layer2_pool"] = nn.MaxPool2d(kernel_size=3, stride=2)
        for i in range(self.blocks[0]):
            layer2[f"layer2_conv{i}"] = Block(ch, 64)
            if i == 0 and hasattr(Block, "expansion"):
                ch = 64
                ch *= Block.expansion
        self.layer2 = Sequential(layer2)

        layer3 = OrderedDict()
        for i in range(self.blocks[1]):
            if i == 0:
                layer3[f"layer3_conv{i}"] = Block(ch, 128, down_sample=True)
                ch = 128
                if hasattr(Block, "expansion"):
                    ch *= Block.expansion
            else:
                layer3[f"layer3_conv{i}"] = Block(ch, 128)
        self.layer3 = Sequential(layer3)

        layer4 = OrderedDict()
        for i in range(self.blocks[2]):
            if i == 0:
                layer4[f"layer4_conv{i}"] = Block(ch, 256, down_sample=True)
                ch = 256
                if hasattr(Block, "expansion"):
                    ch *= Block.expansion
            else:
                layer4[f"layer4_conv{i}"] = Block(ch, 256)
        self.layer4 = Sequential(layer4)

        layer5 = OrderedDict()
        for i in range(self.blocks[3]):
            if i == 0:
                layer5[f"layer5_conv{i}"] = Block(ch, 512, down_sample=True)
                ch = 512
                if hasattr(Block, "expansion"):
                    ch *= Block.expansion
            else:
                layer4[f"layer5_conv{i}"] = Block(ch, 512)
        self.layer5 = Sequential(layer5)

        self.last_channel = ch

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x