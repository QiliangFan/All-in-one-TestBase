from pytorch_lightning import LightningModule
from faster_rcnn import FasterRCNN
from collections import namedtuple
from utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
import torch
from torch import nn
from torch import optim
from torchnet.meter import ConfusionMeter, AverageValueMeter
from torch.nn import functional as F
from typing import Tuple

LossTuple = namedtuple("LossTuple", [
                       "rpn_loc_loss", "rpn_cls_loss", "roi_loc_loss", "roi_cls_loss", "total_loss"])


class Net(LightningModule):

    def __init__(self, faster_rcnn: FasterRCNN):
        super().__init__()

        self.faster_rcnn = faster_rcnn
        self.num_classes = faster_rcnn.n_class

        # 利用gt-bbox将anchor 转为 gt-loc, gt-label
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.rpn_cm = ConfusionMeter(2)  # rpn神经网络score判定的类标: 前景/背景
        # fastrcnn roi的score判定类标: [背景, class1, class2, ..., classN]
        self.roi_cm = ConfusionMeter(self.num_classes+1)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

    def forward(self, imgs: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor, scale):
        n = bboxes.shape[0]
        assert n == 1, "Only support batch size 1"

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(
            features, img_size, scale)

        # since batch_size = 1
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois[0]

        # sample RoIs
        with torch.no_grad():
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi, bbox, label, self.loc_normalize_mean, self.loc_normalize_std)

        # since batch size = 1, there is only one image
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features, sample_roi, sample_roi_index)

        # ----------------------------RPN loss-------------------------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        gt_rpn_label = gt_rpn_label.long()
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, gt_rpn_loc, gt_rpn_label.data, 1)

        rpn_cls_loss = F.cross_entropy(
            rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score[gt_rpn_label > -1]
        self.rpn_cm.add(_rpn_score, _gt_rpn_label)

        # -----------------------------ROI loss------------------------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc: torch.Tensor = roi_cls_loc[torch.arange(
            0, n_sample).long(), gt_roi_label.long()]
        gt_roi_label = gt_roi_label.long()
        gt_roi_loc = gt_roi_loc

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
        )
        roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label)
        self.roi_cm.add(roi_score, gt_roi_label.long())
        
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        """
        batch size只能为1
        """
        imgs, bboxes, labels, scale = batch
        losses = self.forward(imgs, bboxes, labels, scale)
        return losses.total_loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        bacth size只能为1
        """
        imgs, scale = batch
        bboxes, labels, scores = self.faster_rcnn.predict(imgs, imgs.shape)
        return batch_idx

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


def _smooth_l1_loss(x: torch.Tensor, t: torch.Tensor, in_weight: torch.Tensor, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2)) + \
        (1 - flag) * (abs_diff - 0.5 / sigma2)
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc: torch.Tensor, gt_loc: torch.Tensor, gt_label: torch.Tensor, sigma=1):
    with torch.no_grad():
        in_weight = torch.zeros(gt_loc.shape)
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
