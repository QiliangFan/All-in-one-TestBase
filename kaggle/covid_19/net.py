from pytorch_lightning import LightningModule
from collections import namedtuple
from utils.bbox_tools import loc2box
from utils.creator_tool import _get_inside_index

from torch.optim import lr_scheduler
from utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
import torch
from torch import nn
from torch import optim
from torchnet.meter import ConfusionMeter, AverageValueMeter
from torch.nn import functional as F
from typing import Union, Tuple
import cv2
import numpy as np
from visdom_utils import ImageShow
import autogluon.core as ag
from fast_rcnn_vgg import FasterRCNNVGG
from autogluon.core.scheduler.reporter import LocalStatusReporter


LossTuple = namedtuple("LossTuple", [
                       "rpn_loc_loss", "rpn_cls_loss", "roi_loc_loss", "roi_cls_loss", "total_loss"])


class Net(LightningModule):

    def __init__(self, mid_channel = 512, lr = 1e-4, weight_decay = 1e-8, reporter: LocalStatusReporter = None):
        super().__init__()

        # reporter 
        self.reporter = reporter

        self.lr = lr
        self.weight_decay = weight_decay
        self.mid_channel = mid_channel

        self.vis_server = ImageShow()

        faster_rcnn = FasterRCNNVGG(mid_channel=self.mid_channel, n_fg_class=1, ratios=[
                                    0.25, 1, 4], anchor_scales=[4, 8, 16])
        self.faster_rcnn = faster_rcnn
        self.num_classes = faster_rcnn.n_class

        # 利用gt-bbox将anchor 转为 gt-loc, gt-label
        self.anchor_target_creator = AnchorTargetCreator(pos_iou_thresh=0.5)
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.rpn_cm = ConfusionMeter(2)  # rpn神经网络score判定的类标: 前景/背景
        # fastrcnn roi的score判定类标: [class0-others, class1, class2, ..., classN]
        self.roi_cm = ConfusionMeter(self.num_classes)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

        # smooth l1 loss
        self.smooth_loss = nn.SmoothL1Loss(reduction="sum")

    def forward(self, imgs: torch.Tensor, bboxes: torch.Tensor, labels: torch.Tensor, scale: int):
        n = bboxes.shape[0]
        assert n == 1, "Only support batch size 1"

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        # rpn_debug_score 和 rois 是对应的, 用来输出对应roi的score分数
        rpn_locs, rpn_scores, rois, roi_indices, anchor, rpn_debug_score = self.faster_rcnn.rpn(
            features, img_size)

        # since batch_size = 1
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois[0]
        rpn_debug_score: torch.Tensor = rpn_debug_score[0]

        # sample RoIs
        with torch.no_grad():
            # bbox 注意缩放
            _bbox = bbox / scale
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                roi, _bbox, label, self.loc_normalize_mean, self.loc_normalize_std)

        # since batch size = 1, there is only one image
        sample_roi_index = torch.zeros(
            len(sample_roi), dtype=sample_roi.dtype, device=sample_roi.device)
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features, sample_roi, sample_roi_index)

        # ----------------------------RPN loss-------------------------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            _bbox, anchor, img_size)
        gt_rpn_label = gt_rpn_label.long()
        rpn_loc_loss = self.smooth_loss(
            rpn_loc[gt_rpn_label > 0], gt_rpn_loc[gt_rpn_label > 0])
        # rpn_loc_loss = _fast_rcnn_loc_loss(
        #     rpn_loc, gt_rpn_loc, gt_rpn_label.data, 1)

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
        roi_cls_loss += F.binary_cross_entropy_with_logits(roi_score[:, -1], gt_roi_label.float(
        ), pos_weight=torch.as_tensor([4], device=roi_score.device))
        with torch.no_grad():
            self.roi_cm.add(roi_score, gt_roi_label.long())

        losses = [
            rpn_loc_loss,
            rpn_cls_loss * 10,
            roi_loc_loss,
            roi_cls_loss
        ]
        self.log_dict({"rpn_loc": rpn_loc_loss, "rpn_cls": rpn_cls_loss,
                       "roi_loc": roi_loc_loss, "roi_cls": roi_cls_loss}, prog_bar=True)
        losses = losses + [sum(losses)]

        # -----------------------------debug for RPN-----------------------------------#
        # with torch.no_grad():
        #     debug_score_sort = torch.argsort(rpn_debug_score, descending=True)
        #     rpn_debug_score = rpn_debug_score[debug_score_sort][:4]
        #     roi = roi[debug_score_sort][:4]
        #     image = imgs.data
        #     image = self.plot(image, _bbox, color=255)
        #     image = self.plot(image, roi, (0, 255, 0), score=rpn_debug_score)
        #     # image = self.plot(image, sample_roi[gt_roi_label == 0], (255, 0, 0))
        #     self.vis_server.show_image(image)
        # -----------------------------------------------------------------------------#

        # ----------------------------debug for ROI--------------------------------------#
        with torch.no_grad():
            pred_bbox = loc2box(sample_roi, roi_cls_loc[:, -1].view(-1, 4))

            img_H, img_W = imgs.shape[-2], imgs.shape[-1]
            inside_index = _get_inside_index(pred_bbox, img_H, img_W)
            pred_bbox = pred_bbox[inside_index]
            roi_score = roi_score[inside_index]

            argsort = torch.argsort(roi_score[:, -1]).flip(dims=[0])
            sample_roi = sample_roi[argsort]
            roi_cls_loc = roi_cls_loc[argsort]
            pred_bbox = pred_bbox[:8]
            roi_score = roi_score[:, 1][:8]
            img = imgs[0].data
            img = self.plot(img, _bbox)
            img = self.plot(img, pred_bbox, (255, 0, 0), score=roi_score)
            self.vis_server.show_image(img)
        # --------------------------------------------------------------------------------#

        return LossTuple(*losses)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx):
        """
        batch size只能为1
        """
        try:
            imgs, bboxes, labels, scale = batch
            losses = self.forward(imgs, bboxes, labels, scale)
            #     self.vis_server.plot([losses.total_loss],
            #                          batch_idx, "train loss")
            return losses.total_loss
        except:
            import traceback
            traceback.print_exc()
            return None
    
    def training_epoch_end(self, outputs):
        acc = self.compute_accuracy(self.rpn_cm)
        if self.reporter is not None:
            self.reporter(accuracy=acc, epoch=self.current_epoch)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        """
        bacth size只能为1
        """
        imgs, scale = batch
        bboxes, labels, scores = self.faster_rcnn.predict(
            imgs, [imgs.shape[2:]])
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
        opt =  optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_sche = lr_scheduler.LambdaLR(opt, lambda step: self.lr if step > 4000 else self.lr / 1200 + self.lr / 2000 * step)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": lr_sche,
                "interval": "step"
            }
        }

    @staticmethod
    @torch.no_grad()
    def plot(img: Union[torch.Tensor, np.ndarray], bboxes: torch.Tensor, color=255, **kwargs):
        if "score" in kwargs:
            score: torch.Tensor = kwargs["score"]
        else:
            score = None
        if isinstance(img, torch.Tensor):
            img: np.ndarray = img.cpu().squeeze().numpy()
        img = (img - img.min()) / (img.max() - img.min()) * 255
        if isinstance(color, tuple) and img.ndim == 2:
            from PIL import Image
            img = Image.fromarray(img).convert("RGB")
            img = np.asarray(img).astype(np.float32)
        bboxes = bboxes.cpu().numpy().astype(np.int32)
        for i, bbox in enumerate(bboxes):
            if score is not None:
                sc = score[i]
                cv2.putText(img, f"{sc:.2f}", (int(bbox[1]), int(
                    bbox[0]-5)), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 255))
            cv2.rectangle(img, (bbox[1], bbox[0]),
                          (bbox[3], bbox[2]), color=color, thickness=5)
        return img

    def compute_accuracy(self, cm: ConfusionMeter):
        val = cm.value()
        [[tp, fn], [fp, tn]] = val
        return (tp + tn) / (tp + tn + fp + fn)
        

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
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    return loc_loss
