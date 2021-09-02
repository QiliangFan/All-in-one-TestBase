from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from metric import Dice, DiceLoss


class UNet(nn.Module):

    def __init__(self, in_channel = 1):
        super().__init__()

        channel = in_channel * 2
        self.encoder_0 = nn.Conv3d(in_channel, channel, kernel_size=3, stride=1, padding=1)
        

        self.encoder_1 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=2, stride=2, padding=0),
            nn.Conv3d(channel, channel * 2, kernel_size=3, stride=1, padding=1)
        )
        channel *= 2

        self.encoder_2 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=2, stride=2, padding=0),
            nn.Conv3d(channel, channel * 2, kernel_size=3, stride=1, padding=1)
        )
        channel *= 2

        self.encoder_3 = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=2, stride=2, padding=0),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose3d(channel, channel // 2, kernel_size=2, stride=2, padding=0),
        )
        channel //= 2

        self.decoder_3 = nn.Sequential(
            nn.Conv3d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose3d(channel, channel // 2, kernel_size=2, stride=2, padding=0)
        )
        channel //= 2


        self.decoder_2 = nn.Sequential(
            nn.Conv3d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose3d(channel, channel // 2, kernel_size=2, stride=2, padding=0)
        )
        channel //= 2

        self.decoder_1 = nn.Sequential(
            nn.Conv3d(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1),
        )

        self.segment = nn.Sequential(
            nn.Sigmoid()
        )


    def forward(self, x):
        out1 = self.encoder_0(x)
        out2 = self.encoder_1(out1)
        out3 = self.encoder_2(out2)
        out = self.encoder_3(out3)

        out3 = self.decoder_3(torch.cat([out, out3], dim=1))
        out2 = self.decoder_2(torch.cat([out3, out2], dim=1))
        out1 = self.decoder_1(torch.cat([out2, out1], dim=1))

        out = self.segment(x)
        return out

class Net(LightningModule):

    def __init__(self):
        super().__init__()

        self.unet = UNet()

        self.dice = Dice()

        self.dice_loss = DiceLoss()

    def forward(self, x):
        return self.unet(x)
    
    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-4, weight_decay=1e-8)
        return optim

    def training_step(self, batch, batch_idx):
        arr, target = batch
        out = self(arr)
        with torch.no_grad():
            dice = self.dice(out, target)
        dice_loss = self.dice_loss(out, target)
        self.log_dict({
            "dice": dice
        })
        return dice_loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        arr, target = batch
        out = self(arr)
        dice = self.dice(out, target)
        self.log_dict({
            "dice": dice
        })
        