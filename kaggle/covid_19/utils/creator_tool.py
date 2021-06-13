import random

from utils.bbox_tools import loc2box, bbox_iou, bbox2loc
import torch
import torch.nn as nn
from torchvision.ops import nms


class ProposalTargetCreator:
    """根据gt-bbox, 为每个roi生成对应的loc和score

    Returns:
        [type]: [description]
    """

    def __init__(self, n_sample=128, pos_ratio=0.25, pos_iou_thresh=0.5, neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0):
        """[summary]

        Args:
            n_sample (int, optional): The number of sampled regions. Defaults to 128.
            pos_ratio (float, optional): Fractions of regions that is labeled as a foreground. Defaults to 0.25.
            pos_iou_thresh (float, optional): IoU threshold for Roi to be considerer as a foreground. Defaults to 0.5.
            neg_iou_thresh_hi (float, optional): RoI is considered to be the background if IoU is in [neg_iou_thresh_lo, neg_iou_thresh_hi]. Defaults to 0.5.
            neg_iou_thresh_lo (float, optional): [description]. Defaults to 0.0.
        """
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_hi = neg_iou_thresh_hi
        self.neg_iou_thresh_lo = neg_iou_thresh_lo

    def __call__(self, roi: torch.Tensor, bbox: torch.Tensor, label: torch.Tensor, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape

        gt_roi_label = torch.zeros((roi.shape[0],))

        pos_roi_per_image = torch.round(
            torch.as_tensor(self.n_sample * self.pos_ratio))
        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(dim=1)
        max_iou = iou.max(dim=1).values  # 每个roi对应的最大的IOU

        pos_index = torch.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(
            min(pos_roi_per_image.item(), len(pos_index)))
        if len(pos_index) > 0:
            pos_index = random.sample(
                pos_index.tolist(), pos_roi_per_this_image)
        gt_roi_label[pos_index] = 1

        neg_index = torch.where((max_iou < self.neg_iou_thresh_hi) & (
            self.neg_iou_thresh_lo <= max_iou))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        if len(neg_index) > 0:
            neg_index = random.sample(
                neg_index.tolist(), neg_roi_per_this_image)
        gt_roi_label[neg_index] = 1

        keep_index = pos_index + neg_index
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # 0 -> background
        sample_roi = roi[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc - torch.as_tensor(loc_normalize_mean, dtype=torch.float32, device=gt_roi_loc.device)
                      ) / torch.as_tensor(loc_normalize_std, dtype=torch.float32, device=gt_roi_loc.device)

        return sample_roi, gt_roi_loc, gt_roi_label


class ProposalCreator:
    def __init__(self,
                 parent_model: nn.Module,
                 nms_thresh: float = 0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,  # NMS之后保留的bounding box数量
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16,  # bounding box的最小尺寸
                 ):
        super().__init__()
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc: torch.Tensor, score: torch.Tensor, anchor: torch.Tensor, img_size, scale=1.):
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2box(anchor, loc)
        roi[:, 0:4:2] = torch.clip(roi[:, 0:4:2], 0, img_size[0])
        roi[:, 1:4:2] = torch.clip(roi[:, 1:4:2], 0, img_size[1])

        min_size = self.min_size * scale
        hs = roi[:, 3] - roi[:, 1]
        ws = roi[:, 2] - roi[:, 0]
        keep = torch.where((hs > min_size) & (ws > min_size))[0]
        roi, score = roi[keep, :], score[keep]

        order = score.flatten().argsort().flip(dims=[0])  # largs -> small
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi, score = roi[order, :], score[order]

        keep = nms(roi, score, self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi


class AnchorTargetCreator:
    def __init__(self, n_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, pos_ratio=0.5):
        """[summary]

        Args:
            n_sample (int, optional): The number of regions to produce. Defaults to 256.
            pos_iou_thresh (float, optional): . Defaults to 0.7.
            neg_iou_thresh (float, optional): [description]. Defaults to 0.3.
            pos_ratio (float, optional): Ratio of positive regions in the sampled regions. Defaults to 0.5.
        """
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox: torch.Tensor, anchor: torch.Tensor, img_size):
        """[summary]

        Args:
            bbox (torch.Tensor): shape (R, 4)
            anchor (torch.Tensor): shape (S, 4)
            img_size ([type]): [description]
        """
        img_H, img_W = img_size

        n_anchor = len(anchor)
        inside_index = _get_inside_index(anchor, img_H, img_W)
        anchor = anchor[inside_index]
        argmax_ious, label = self._create_label(inside_index, anchor, bbox)

        loc = bbox2loc(anchor, bbox[argmax_ious])

        label = _unmap(label, n_anchor, inside_index, fill=-1)
        loc = _unmap(loc, n_anchor, inside_index, fill=0)

        return loc, label

    def _create_label(self, inside_index, anchor: torch.Tensor, bbox: torch.Tensor):
        # label: 1 is positive, 0 is negative, -1 is don't care
        label = torch.empty((len(inside_index),), dtype=torch.int32)
        label.fill_(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(
            anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0
        label[gt_argmax_ious] = 1  # 每个bbox最大的anchor
        label[max_ious >= self.pos_iou_thresh] = 1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = torch.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = random.sample(
                pos_index.tolist(), len(pos_index) - n_pos)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - torch.sum(label == 1)
        neg_index = torch.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = random.sample(
                neg_index.tolist(), len(neg_index) - n_neg)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor: torch.Tensor, bbox: torch.Tensor, inside_index):
        ious = bbox_iou(anchor, bbox)

        # 为每个anchor分配一个ground truth bbox
        argmax_ious = ious.argmax(dim=1)
        # 对每个anchor, 与之iou最大的bbox
        max_ious = ious[range(len(inside_index)), argmax_ious]

        # 排个序
        gt_argmax_ious = ious.argmax(dim=0)
        gt_max_ious = ious[gt_argmax_ious, range(
            len(ious.shape[1]))]  # 获取与bbox相对应IOU最大的anchor
        gt_argmax_ious = torch.where(ious == gt_max_ious)[
            0]  # 获取与bbox绑定了的anchor的索引

        return argmax_ious, max_ious, gt_argmax_ious


def _get_inside_index(anchor: torch.Tensor, H, W):
    """获取在原图范围内的anchor的第一维索引

    Args:
        anchor (torch.Tensor): [description]
        H ([type]): [description]
        W ([type]): [description]

    Returns:
        [type]: [description]
    """
    index_inside = torch.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside


def _unmap(data: torch.Tensor, count, index, fill=0):
    if len(data.shape) == 1:
        ret = torch.empty((count, ), dtype=data.dtype)
        ret.fill_(fill)
        ret[index] = data
    else:
        ret = torch.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill_(fill)
        ret[index, :] = data
    return ret
