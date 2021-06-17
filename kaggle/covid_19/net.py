from utils.bbox_tools import loc2box
from pytorch_lightning import LightningModule
from faster_rcnn import FasterRCNN
from collections import namedtuple
from utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
import torch
from torch import nn
from torch import optim
from torchnet.meter import ConfusionMeter, AverageValueMeter
from torch.nn import functional as F
from typing import Union, Tuple, cast
import cv2
import numpy as np
from visdom_utils import ImageShow
from scipy.ndimage import zoom

LossTuple = namedtuple("LossTuple", [
                       "rpn_loc_loss", "rpn_cls_loss", "roi_loc_loss", "roi_cls_loss", "total_loss"])


class Net(LightningModule):

    def __init__(self, faster_rcnn: FasterRCNN):
        super().__init__()

        self.vis_server = ImageShow()

        self.faster_rcnn = faster_rcnn
        self.num_classes = faster_rcnn.n_class

        # 利用gt-bbox将anchor 转为 gt-loc, gt-label
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.rpn_cm = ConfusionMeter(2)  # rpn神经网络score判定的类标: 前景/背景
        # fastrcnn roi的score判定类标: [class0-others, class1, class2, ..., classN]
        self.roi_cm = ConfusionMeter(self.num_classes)
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
        sample_roi_index = torch.zeros(len(sample_roi), dtype=sample_roi.dtype, device=sample_roi.device)
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features, sample_roi, sample_roi_index)

        # ----------------------------RPN loss-------------------------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            bbox, anchor, img_size)
        gt_rpn_label = gt_rpn_label.long()
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc, gt_rpn_loc, gt_rpn_label.data, 1)

        rpn_cls_loss = F.cross_entropy(
            rpn_score, gt_rpn_label, ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = rpn_score[gt_rpn_label > -1]
        with torch.no_grad():
            self.rpn_cm.add(_rpn_score, _gt_rpn_label)

        # -----------------------------ROI loss------------------------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc: torch.Tensor = roi_cls_loc[torch.arange(
            0, n_sample).long(), gt_roi_label.long()]
        gt_roi_label = gt_roi_label.long()

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
        )
        roi_cls_loss = F.cross_entropy(roi_score, gt_roi_label)
        with torch.no_grad():
            self.roi_cm.add(roi_score, gt_roi_label.long())
        
        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        # pred for test
        with torch.no_grad():
            argsort = torch.argsort(roi_score[:, 1]).flip(dims=[0])[:8]
            sample_roi = sample_roi[argsort]
            roi_cls_loc = roi_cls_loc[argsort]
            pred_bbox = loc2box(sample_roi, roi_cls_loc[:, 1].view(-1, 4))
            img_H, img_W = imgs.shape[-2], imgs.shape[-1]
            tmp = torch.ones_like(pred_bbox, device=pred_bbox.device)
            tmp[:, 0::2] = img_H
            tmp[:, 1::2] = img_W
            pred_bbox = pred_bbox[torch.where((0 <= pred_bbox) & (pred_bbox < tmp))[0]]

        return LossTuple(*losses), pred_bbox

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        """
        batch size只能为1
        """
        try:
            imgs, bboxes, labels, scale = batch
            losses, pred_bbox = self.forward(imgs, bboxes, labels, scale)
            with torch.no_grad():
                bboxes, labels = bboxes[0], labels[0]
                bboxes /= scale
                pred_bbox /= scale

                img = imgs[0].data
                img = self.plot(img, bboxes)
                img = self.plot(img, pred_bbox, color=10)
                self.vis_server.show_text(str(pred_bbox.tolist()))
                self.vis_server.show_image(img)
            return losses.total_loss
        except:
            import traceback
            traceback.print_exc()
            return None

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        bacth size只能为1
        """
        imgs, scale = batch
        bboxes, labels, scores = self.faster_rcnn.predict(imgs, [imgs.shape[2:]])
        # since batch size = 1
        bboxes, labels, scores = bboxes[0], labels[0], scores[0]
        argsort = torch.argsort(scores).flip(dims=[0])
        bboxes = bboxes[argsort[:32]]
        labels = labels[argsort[:32]]
        scores = scores[argsort[:32]]
        img = self.plot(imgs[0], bboxes, batch_idx)
        self.vis_server.show_image(img)
        return batch_idx

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    @staticmethod
    @torch.no_grad()
    def plot(img: Union[torch.Tensor, np.ndarray], bboxes: torch.Tensor, color: int = 255):
        if isinstance(img, torch.Tensor):
            img: np.ndarray = img.cpu().squeeze().numpy()
        if 0 <= img.max() <= 1:
            img *= 255
        if min(img.shape) > 256:
                bboxes[:, ::2] *= 256/img.shape[0]
                bboxes[:, 1::2] *= 256/img.shape[1]
                img = zoom(img, (256/img.shape[0], 256/img.shape[1]), mode="nearest")
        for bbox in bboxes:
            bbox = bbox.cpu().numpy().astype(np.int32)
            cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=color, thickness=2)
        return img


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
        in_weight = torch.zeros(gt_loc.shape, device=gt_loc.device)
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    return loc_loss
