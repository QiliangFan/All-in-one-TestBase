from typing import cast
from pytorch_lightning import LightningModule
from torch.optim import lr_scheduler
import torch
from torch import nn
from torch.optim import Adam, SGD
from metric import Dice, DiceLoss
from visdom import Visdom

def conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm3d(channel, momentum=0.2),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm3d(channel, momentum=0.2),
            # nn.ReLU()
        )

def transpose_conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, output_padding=output_padding),
        nn.InstanceNorm3d(channel, momentum=0.2),
        nn.ReLU()
    )



class UNet(nn.Module):

    def __init__(self, in_channel = 1):
        super().__init__()

        expand = 2

        channel = in_channel * expand
        self.encoder_0 = nn.Sequential(
            conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1),
            conv_block(channel, channel, kernel_size=3, stride=1, padding=1)
        )

        self.encoder_1 = nn.Sequential(
            conv_block(channel, channel, kernel_size=2, stride=2, padding=0),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            conv_block(channel, channel * expand, kernel_size=3, stride=1, padding=1),
        )
        channel *= expand

        self.encoder_2 = nn.Sequential(
            conv_block(channel, channel, kernel_size=2, stride=2, padding=0),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            conv_block(channel, channel * expand, kernel_size=3, stride=1, padding=1),
        )
        channel *= expand

        self.encoder_3 = nn.Sequential(
            conv_block(channel, channel, kernel_size=2, stride=2, padding=0),
            # nn.MaxPool3d(kernel_size=2, stride=2),
            # conv_block(channel, channel, kernel_size=3, stride=1, padding=1),
            # transpose_conv_block(channel, channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            transpose_conv_block(channel, channel, kernel_size=2, stride=2, padding=0, output_padding=0),
        )

        self.decoder_3 = nn.Sequential(
            conv_block(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            # conv_block(channel, channel, kernel_size=3, stride=1, padding=1),
            # transpose_conv_block(channel, channel // expand, kernel_size=3, stride=2, padding=1, output_padding=1),
            transpose_conv_block(channel, channel // expand, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        channel //= expand


        self.decoder_2 = nn.Sequential(
            conv_block(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            # conv_block(channel, channel, kernel_size=3, stride=1, padding=1),
            # transpose_conv_block(channel, channel // expand, kernel_size=3, stride=2, padding=1, output_padding=1),
            transpose_conv_block(channel, channel // expand, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        channel //= expand

        self.decoder_1 = nn.Sequential(
            conv_block(channel * 2, channel, kernel_size=3, stride=1, padding=1, relu=False),
            conv_block(channel, channel // expand, kernel_size=3, stride=1, padding=1, relu=False),
            conv_block(channel // expand, channel // expand, kernel_size=3, stride=1, padding=1, relu=False),
        )
        channel //= expand

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

        out = self.segment(out1)
        return out

class Net(LightningModule):

    def __init__(self):
        super().__init__()

        self.unet = UNet()

        self.dice = Dice()

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()

        self.vis = Visdom(port=8888)

    def forward(self, x):
        return self.unet(x)
    
    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-3, weight_decay=1e-12)
        # lr_sche = StepLR(optim, step_size=10, gamma=0.9)
        lr_sche = lr_scheduler.ReduceLROnPlateau(optim, mode="max", verbose=True)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_sche,
                "monitor": "dice"
            }
        }

    def training_step(self, batch, batch_idx):
        arr, target = batch
        out = self(arr)
        self.show_out(target, "target")
        self.show_out(out)
        dice = self.dice(out, target)
        self.log_dict({
            "dice": dice
        }, prog_bar=True, on_epoch=False, on_step=True)
        # if self.trainer.current_epoch < 50:
        #     loss = self.ce_loss(out, target) + self.dice_loss(out, target)
        # else:
        #     loss = self.dice_loss(out, target)
        # loss = self.ce_loss(out, target)
        loss = self.dice_loss(out, target)
        # loss = self.ce_loss(out, target) + self.dice_loss(out, target) * 2

        # return dice_loss
        return loss

    def test_step(self, batch, batch_idx):
        arr, target = batch
        out = self(arr)
        dice = self.dice(out, target)
        self.log_dict({
            "dice": dice
        }, prog_bar=True)
        return batch_idx

    @torch.no_grad()
    def show_out(self, out, name = "out"):
        for i, arr in enumerate(out):
            arr = cast(torch.Tensor, arr)
            arr = arr.squeeze(dim=0).cpu().detach().numpy()
            slice = arr[31]
            self.vis.heatmap(slice, win=f"{name}_{i}")
    
