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
            nn.BatchNorm3d(channel),
            nn.ReLU()
        )
    else:
        return nn.Sequential(
            nn.Conv3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(channel),
            # nn.ReLU()
        )

def transpose_conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1, output_padding=1):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channel, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True, output_padding=output_padding),
        nn.BatchNorm3d(channel),
        nn.ReLU()
    )



class UNet(nn.Module):

    def __init__(self, in_channel = 1, visdom: Visdom = None):
        super().__init__()

        self.idx = 0

        self.vis = visdom

        expand = 4

        channel = in_channel * expand
        self.encoder_0 = nn.Sequential(
            conv_block(in_channel, channel, kernel_size=3, stride=1, padding=1),
            conv_block(channel, channel, kernel_size=3, stride=1, padding=1),
        )

        self.encoder_1 = nn.Sequential(
            conv_block(channel, channel, kernel_size=2, stride=2, padding=0),
            conv_block(channel, channel * expand, kernel_size=3, stride=1, padding=1),
            conv_block(channel * expand, channel * expand, kernel_size=3, stride=1, padding=1),
        )
        channel *= expand

        self.encoder_2 = nn.Sequential(
            conv_block(channel, channel, kernel_size=2, stride=2, padding=0),
            conv_block(channel, channel * expand, kernel_size=3, stride=1, padding=1),
            conv_block(channel * expand, channel * expand, kernel_size=3, stride=1, padding=1),
        )
        channel *= expand

        self.encoder_3 = nn.Sequential(
            conv_block(channel, channel, kernel_size=2, stride=2, padding=0),
            conv_block(channel, channel, kernel_size=3, stride=1, padding=1),
            conv_block(channel, channel, kernel_size=3, stride=1, padding=1),
            conv_block(channel, channel, kernel_size=3, stride=1, padding=1),
            transpose_conv_block(channel, channel, kernel_size=2, stride=2, padding=0, output_padding=0),
        )

        self.decoder_3 = nn.Sequential(
            conv_block(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1),
            conv_block(channel * 2, channel, kernel_size=3, stride=1, padding=1),
            # conv_block(channel, channel, kernel_size=3, stride=1, padding=1, relu=True),

            conv_block(channel, channel  // expand, kernel_size=3, stride=1, padding=1),
            transpose_conv_block(channel // expand, channel // expand, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        channel //= expand


        self.decoder_2 = nn.Sequential(
            conv_block(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1, relu=True),
            conv_block(channel * 2, channel, kernel_size=3, stride=1, padding=1, relu=True),
            # conv_block(channel, channel, kernel_size=3, stride=1, padding=1, relu=True),
            conv_block(channel, channel  // expand, kernel_size=3, stride=1, padding=1, relu=True),
            transpose_conv_block(channel // expand, channel // expand, kernel_size=2, stride=2, padding=0, output_padding=0),
        )
        channel //= expand

        self.decoder_1 = nn.Sequential(
            conv_block(channel * 2, channel * 2, kernel_size=3, stride=1, padding=1, relu=True),
            conv_block(channel * 2, channel, kernel_size=3, stride=1, padding=1, relu=True),
            # conv_block(channel, channel, kernel_size=3, stride=1, padding=1, relu=True),
            conv_block(channel, channel  // expand, kernel_size=3, stride=1, padding=1, relu=True),
            conv_block(channel // expand, channel // expand, kernel_size=3, stride=1, padding=1, relu=True),
        )
        channel //= expand

        self.segment = nn.Sequential(
            nn.Conv3d(channel, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        out1 = self.encoder_0(x)
        out2 = self.encoder_1(out1)
        out3 = self.encoder_2(out2)
        out = self.encoder_3(out3)

        if self.vis is not None and self.idx % 4 == 0:
            self.vis.heatmap(out1[0][-1][8], "out1")
            self.vis.heatmap(out2[0][-1][8], "out2")
            self.vis.heatmap(out3[0][-1][8], "out3")
            self.vis.heatmap(out[0][-1][8], "out")

        out3 = self.decoder_3(torch.cat([out, out3], dim=1))
        out2 = self.decoder_2(torch.cat([out3, out2], dim=1))
        out1 = self.decoder_1(torch.cat([out2, out1], dim=1))

        out = self.segment(out1)

        if self.vis is not None and self.idx % 4 == 0:
            self.vis.heatmap(out1[0][-1][8], "_out1")
            self.vis.heatmap(out2[0][-1][8], "_out2")
            self.vis.heatmap(out3[0][-1][8], "_out3")
            self.vis.heatmap(out[0][-1][8], "_out")

        self.idx += 1
        return out

class Net(LightningModule):

    def __init__(self):
        super().__init__()
        self.vis = Visdom(port=8888)

        self.unet = UNet()
        # self.unet = UNet(visdom=self.vis)

        self.dice = Dice()

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()


    def forward(self, x):
        return self.unet(x)
    
    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=1e-3, weight_decay=1e-6)
        # lr_sche = StepLR(optim, step_size=10, gamma=0.9)
        lr_sche = lr_scheduler.MultiStepLR(optim, milestones=[30, 60, 90, 120, 150, 180], gamma=0.5)
        return {
            "optimizer": optim,
            # "lr_scheduler": {
            #     "scheduler": lr_sche,
            #     "monitor": "dice"
            # }
        }

    def training_step(self, batch, batch_idx):
        arr, target = batch
        out = self(arr)
        if batch_idx % 4 == 0:
            self.show_out(out)
            # self.show_out(arr, "arr")
        dice = self.dice(out, target)
        self.log_dict({
            "dice": dice
        }, prog_bar=True, on_epoch=False, on_step=True)
        # if self.trainer.current_epoch < 50:
        #     loss = self.ce_loss(out, target) + self.dice_loss(out, target)
        # else:
        #     loss = self.dice_loss(out, target)
        # loss = self.ce_loss(out, target)
        # loss = self.dice_loss(out, target)
        # cur_epoch = self.trainer.current_epoch
        # if cur_epoch < 200:
        #     ce_co, dice_co = 0.2, 1
        # elif 200 <= cur_epoch < 400:
        #     ce_co, dice_co = 1, 1
        # else:
        #     ce_co, dice_co = 2, 1

        # loss = ce_co * self.ce_loss(out, target) + dice_co * self.dice_loss(out, target)
        loss = self.ce_loss(out, target) + self.dice_loss(out, target)

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
    
