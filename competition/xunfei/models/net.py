from typing import Tuple, Sequence, cast
import cv2
import numpy as np
import torch
from visdom import Visdom
from torch import nn
from pytorch_lightning import LightningModule
from .resnet import ResNet
from .rpn import RPN

class Net(LightningModule):
    win_idx = 0

    def __init__(self, 
        num_classes: int, 
        labels: Sequence[str],
        anchor_ratios =  [0.5, 1, 2], 
        anchor_size = [4, 8, 16],
        min_size = 4):
        """
        num_classes: fg_class + 1 (bg_class: 0)
        """
        super().__init__()
        self.init_viddom()

        self.num_classes = num_classes
        self.labels = labels
        self.anchor_ratios = anchor_ratios
        self.anchor_size = anchor_size
        self.min_size = min_size

        # anchor prepare works
        self.n_anchor = len(self.anchor_ratios) * len(self.anchor_size)
        anchor_base = []
        for ratio in self.anchor_ratios:
            for size in self.anchor_size:
                weight = size
                height = size * ratio
                anchor_base.append([0, 0, height, weight])
        self.anchor_base = torch.as_tensor(anchor_base)

        self.feature = ResNet(layers=18)
        self.feat_stride = 16

        self.rpn = RPN(in_channel=self.feature.last_channel, num_classes=num_classes, n_anchor=self.n_anchor)

        self.dpn_cls_loss = nn.CrossEntropyLoss()
        self.dpn_loc_loss = nn.SmoothL1Loss()


    def forward(self, img: torch.Tensor, gt: torch.Tensor, img_size: Tuple[int, int]):
        n_batch = img.shape[0]
        assert n_batch == 1

        # Ground Truth
        gt_bbox = gt[:, :4]
        gt_label = gt[:, 4]
        
        features: torch.Tensor = self.feature(img)

        anchors = self.generate_anchors((features.shape[-2], features.shape[-1])).to(device=img.device)

        # roi: (x1, y1, x2, y2)
        out, cls_softmax, loc, roi, roi_score, roi_label = self.rpn(features, anchors, self.feat_stride, img_size)

        # ------------------------------- RPN ---------------------------------------------------
        # assign labels corresponding to rois... 
        # compute loss: anchor -> roi


        # DEBUG
        self.show_image(img, roi, roi_score, roi_label)


        # ---------------------------------------------------------------------------------------------
        # compute loss: roi -> gt_bbox (可见roi是多么重要)

        # ------------------------------- roi ---------------------------------------------------

        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        arr, bboxs = batch
        self(arr, img_size=arr.shape[2:])

    def test_step(self, batch, batch_idx: int):
        arr, bboxs = batch
        self(arr, img_size=arr.shape[2:])

    def generate_anchors(self, feat_size: Tuple[int, int]):
        H, W = feat_size
        shift_y = torch.arange(0, H*self.feat_stride, self.feat_stride)
        shift_x = torch.arange(0, W*self.feat_stride, self.feat_stride)
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
        shift_x, shift_y = shift_x.flatten(), shift_y.flatten()
        shift = torch.stack((shift_y, shift_x, shift_y, shift_x), dim=1)  # (N, 4)

        num_anchor_base = self.anchor_base.shape[0]
        num_shift = shift.shape[0]

        anchors = shift[:, None] + self.anchor_base[None, :]  # (H*W, n_anchor, 4)
        anchors = anchors.view(H, W, self.n_anchor, 4)

        return anchors

    def init_viddom(self):
        self.vis = Visdom(env="xunfei")

    @torch.no_grad()
    def show_image(self, img, box, score, labels):
        # img: (C, H, W) & (RGB)
        img, box, score, labels = img.cpu().numpy().squeeze(), box.cpu().numpy(), score.cpu().numpy(), labels.cpu().numpy()
        img = cast(np.ndarray, img)
        img = img.transpose([1, 2, 0])
        img = np.flip(img, axis=2)

        arr = img.copy()
        for x1, y1, x2, y2 in box[:8]:
            cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
            # cv2.putText()

        arr = torch.as_tensor(arr.transpose([2, 0, 1]))
        arr = torch.flip(arr, dims=[0])
        self.vis.image(arr, win=f"image_{self.win_idx % 6}")
        self.win_idx += 1