from typing import Sequence, Tuple, cast

import cv2
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from visdom import Visdom

from .resnet import ResNet
from .roi import ROIHead
from .rpn import RPN


class Net(LightningModule):
    idx = 0

    def __init__(self,
                 num_classes: int,
                 labels: Sequence[str],
                 anchor_ratios=[0.5, 1, 2],
                 anchor_size=[4, 8, 16],
                 ):
        """
        num_classes: fg_class + 1 (bg_class: 0)
        """
        super().__init__()
        self.init_viddom()

        self.num_classes = num_classes
        self.labels = labels
        self.anchor_ratios = anchor_ratios
        self.anchor_size = anchor_size

        # anchor prepare works
        self.n_anchor = len(self.anchor_ratios) * len(self.anchor_size)
        anchor_base = []
        for ratio in self.anchor_ratios:
            for size in self.anchor_size:
                height = size
                width = size * ratio
                anchor_base.append(
                    [-0.5 * height, -0.5 * width, 0.5 * height, 0.5 * width])
        self.anchor_base = torch.as_tensor(anchor_base)

        self.feature = ResNet(layers=18)
        self.feat_stride = 16

        self.rpn = RPN(in_channel=self.feature.last_channel,
                       num_classes=num_classes, n_anchor=self.n_anchor)

        self.roi_head = ROIHead(
            self.feature.last_channel * RPN.roi_pool_size * RPN.roi_pool_size,
            num_classes
        )

        self.anchor_cls_loss = nn.CrossEntropyLoss()
        self.anchor_loc_loss = nn.SmoothL1Loss()

        self.roi_cls_loss = nn.CrossEntropyLoss()
        self.roi_loc_loss = nn.SmoothL1Loss()

    def forward(self, img: torch.Tensor, img_size: Tuple[int, int], gt: torch.Tensor = None):
        n_batch = img.shape[0]
        assert n_batch == 1

        if self.training:
            gt = cast(torch.Tensor, gt)
            # Ground Truth (only support batch size = 1)
            gt_bbox = gt[0, :, :4]
            gt_label = gt[0, :, 4]

            _gt_bbox = torch.stack([
                gt_bbox[:, 1],
                gt_bbox[:, 0],
                gt_bbox[:, 1] + gt_bbox[:, 3],
                gt_bbox[:, 0] + gt_bbox[:, 2]
            ], dim=1)
        else:
            gt_bbox = None
            gt_label = None

            _gt_bbox = None

        features: torch.Tensor = self.feature(img)

        anchors = self.generate_anchors(
            (features.shape[-2], features.shape[-1])).to(device=img.device)

        # roi: (x1, y1, x2, y2)
        out, cls_softmax, loc, roi, roi_score, roi_label, gt_anchor_loc, gt_anchor_label = self.rpn(
            features, anchors, self.feat_stride, img_size, gt_bbox, gt_label)
        if out is None or roi.shape[0] == 0:
            return None

        roi_loc, roi_cls, pred_box, pred_box_score, gt_roi_loc, gt_roi_label = self.roi_head(
            out, roi, img_size, gt_anchor_label if self.training else roi_label, gt_bbox, gt_label)

        # ------------------------------- RPN (anchor) ---------------------------------------------------
        # assign labels corresponding to rois...
        # compute loss: anchor -> roi
        # DEBUG
        if _gt_bbox is not None:
            self.show_image(img, _gt_bbox, win=f"{'training' if self.training else 'test'}gt")
        self.show_image(img, roi, labels=gt_anchor_label, win=f"{'training' if self.training else 'test'}roi")

        # ---------------------------------------------------------------------------------------------

        # ------------------------------- roi ---------------------------------------------------
        # compute loss: roi -> gt_bbox (可见roi是多么重要)
        # DEBUG
        _pred_box = torch.stack([pred_box[:, 1], pred_box[:, 0], pred_box[:, 3] +
                                 pred_box[:, 1], pred_box[:, 2]+pred_box[:, 0]], dim=1)
        _pred_box = _pred_box[:8]
        self.show_image(img, _pred_box, win=f"{'training' if self.training else 'test'}pred_bbox")
        # ----------------------------------------------------------------------------------------

        if self.training:
            anchor_loc_loss = torch.nan_to_num(self.anchor_loc_loss(
                loc[gt_anchor_label > 0], gt_anchor_loc[gt_anchor_label > 0]), posinf=10)
            anchor_cls_loss = self.anchor_cls_loss(
                cls_softmax, gt_anchor_label.long())
            roi_loc_loss = torch.nan_to_num(self.roi_loc_loss(
                roi_loc[gt_roi_label > 0], gt_roi_loc[gt_roi_label > 0]), posinf=10)
            roi_cls_loss = self.roi_cls_loss(roi_cls, gt_roi_label.long())
            self.log_dict({
                "anchor_loc_loss": anchor_loc_loss,
                "anchor_cls_loss": anchor_cls_loss,
                "roi_loc_loss": roi_loc_loss,
                "roi_cls_loss": roi_cls_loss
            }, prog_bar=True)
            return anchor_loc_loss + anchor_cls_loss + roi_loc_loss + roi_cls_loss
        else:
            pred_box = pred_box[torch.where(pred_box_score > 0.5)[0]]
            pred_box_score = pred_box_score[torch.where(pred_box_score > 0.5)[0]]
            return pred_box, pred_box_score

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-8)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        arr, bboxs = batch
        loss = self(arr, img_size=arr.shape[2:], gt=bboxs)
        return loss

    def test_step(self, batch, batch_idx: int):
        arr, bboxs = batch
        self(arr, img_size=arr.shape[2:], gt=bboxs)

    def generate_anchors(self, feat_size: Tuple[int, int]):
        H, W = feat_size
        shift_y = torch.arange(0, H*self.feat_stride, self.feat_stride)
        shift_x = torch.arange(0, W*self.feat_stride, self.feat_stride)
        shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
        shift_x, shift_y = shift_x.flatten(), shift_y.flatten()
        shift = torch.stack(
            (shift_y, shift_x, shift_y, shift_x), dim=1)  # (N, 4)

        num_anchor_base = self.anchor_base.shape[0]
        num_shift = shift.shape[0]

        anchors = shift[:, None] + \
            self.anchor_base[None, :]  # (H*W, n_anchor, 4)
        anchors = anchors.view(H, W, self.n_anchor, 4)   # (y1, x1, y2, x2)
        anchors = torch.stack([anchors[:, :, :, 0], anchors[:, :, :, 1], anchors[:, :,
                                                                                 :, 2]-anchors[:, :, :, 0], anchors[:, :, :, 3]-anchors[:, :, :, 1]], dim=3)

        return anchors

    def init_viddom(self):
        self.vis = Visdom(env="xunfei")

    @torch.no_grad()
    def show_image(self, img, box, win: str="default", labels=None):
        """
        box: (x1, y1, x2, y2) 
        """
        # img: (C, H, W) & (RGB)
        img, box = img.cpu().numpy().squeeze(), box.cpu().numpy()
        img = cast(np.ndarray, img)
        img = img.transpose([1, 2, 0])

        if labels is not None:
            labels = labels.cpu().numpy()
            fg_idx = np.where(labels > 0)[0]
            bg_idx = np.where(labels == 0)[0]
        else:
            fg_idx = list(range(box.shape[0]))
            bg_idx = []

        arr = img.copy()
        for x1, y1, x2, y2 in box[fg_idx]:
            if win.endswith("pred_bbox"):
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)
            cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)),
                          color=color, thickness=2, lineType=cv2.LINE_AA)
            # cv2.putText()
        for x1, y1, x2, y2 in box[bg_idx]:
            cv2.rectangle(arr, (int(x1), int(y1)), (int(x2), int(y2)), color=(
                0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        arr = torch.as_tensor(arr.transpose([2, 0, 1]))
        arr = torch.flip(arr, dims=[0])
        self.vis.image(
            arr, win=f"{win}{self.idx % 2}")
        if win.endswith("pred_bbox"):
            self.idx += 1
