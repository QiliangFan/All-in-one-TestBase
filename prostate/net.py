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

    def __init__(self, visdom=True):
        super().__init__()
        if visdom:
            self.vis = Visdom(port=8888)
        else:
            self.vis = None

        self.unet = UNet()
        # self.unet = UNet(visdom=self.vis)

        self.dice = Dice()

        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCEWithLogitsLoss()

        self.lr = 1e-3

    def forward(self, x):
        return self.unet(x)
    
    def configure_optimizers(self):
        optim = Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        # lr_sche = StepLR(optim, step_size=10, gamma=0.9)
        # lr_sche = lr_scheduler.StepLR(optim, 100, gamma=0.5)
        lr_sche = lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.9, patience=20)
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_sche,
                "monitor": "dice"
            }
        }

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        # warm up
        if self.trainer.global_step < 200:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 200.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr
        
        # lr reduce
        if self.trainer.current_epoch % 1000 == 0 and self.trainer.current_epoch <= 2000:
            for pg in optimizer.param_groups:
                pg["lr"] = self.lr

        optimizer.step(closure=optimizer_closure)
            

    def training_step(self, batch, batch_idx):
        arr, target = batch
        out = self(arr)
        if batch_idx % 4 == 0:
            self.show_out(out)
            # self.show_out(arr, "arr")
        dice = self.dice(out, target)
        
        cur_epoch = self.trainer.current_epoch

        if cur_epoch <= 1000:
            loss = self.dice_loss(out, target)
        elif cur_epoch <= 2000:
            loss = self.ce_loss(out, target)
        else:
            loss = self.ce_loss(out, target) + self.dice_loss(out, target)

        self.log_dict({
            "dice": dice,
            "lr": self.optimizers().param_groups[0]['lr']
        }, prog_bar=True, on_epoch=False, on_step=True)

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
        if self.vis is None:
            return
        for i, arr in enumerate(out):
            arr = cast(torch.Tensor, arr)
            arr = arr.squeeze(dim=0).cpu().detach().numpy()
            slice = arr[31]
            self.vis.heatmap(slice, win=f"{name}_{i}")
    
