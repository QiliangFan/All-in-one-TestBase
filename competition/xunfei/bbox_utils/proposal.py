from typing import Tuple
from six import class_types
import torch
import numpy as np
from .utils import bbox_iou, loc2box, get_inside_index, box2loc
from torchvision.ops import nms


class ProposalCreator:

    def __init__(self, num_post_nms: int = 3000, nms_threshold: float = 0.7, min_size=4):
        """[summary]

        Args:
            num_post_nms (int): The number of remained samples after NMS
            nms_threshold (float): nms_threshold
        """
        self.num_post_nms = num_post_nms
        self.num_post_nms_test = num_post_nms // 50
        self.nms_threshold = nms_threshold
        self.min_size = min_size

    def __call__(self, score: torch.Tensor, loc: torch.Tensor, anchor: torch.Tensor, img_size: Tuple[int, int], gt_bbox: torch.Tensor = None, gt_label: torch.Tensor = None):
        """[summary]

        Args:
            score (torch.Tensor): [n_batch, H, W, n_anchor, num_class]
            loc (torch.Tensor): [n_batch, H, W, n_anchor, 4]
            anchor (torch.Tensor): [H, W, n_anchor, 4]
            gt_bbox: [N, 4]
            gt_label: [N]
        """
        n_batch = score.shape[0]
        assert n_batch == 1

        H, W = img_size

        score = score[0]
        loc = loc[0]

        # loc = (loc - loc.mean(dim=[2, 3])[:, :, None, None])
        # loc = (loc) / loc.std()
        loc = (loc - loc.mean()) / loc.std()
        # loc = (loc - loc.mean(dim=[2, 3])[:, :, None, None]) / loc.std(dim=[2, 3])[:, :, None, None]

        # roi
        roi = loc2box(anchor, loc)
        roi = roi.view(-1, 4).contiguous()  # (y, x, h, w)

        # inside index
        roi[:, 2] = roi[:, 0] + roi[:, 2]
        roi[:, 3] = roi[:, 1] + roi[:, 3]
        roi = roi[:, [1, 0, 3, 2]]  # (x1, y1, x2, y2)
        # inside_index = list(range(roi.shape[0]))
        inside_index = get_inside_index(roi, H, W)
        roi = roi[inside_index]
        anchor = anchor.view(-1, 4).contiguous()
        anchor = anchor[inside_index]

        # cls & NMS
        cls_score, cls_label = score.max(dim=-1)  # [H, W, n_anchor]
        cls_label = cls_label.view(-1)
        cls_score = cls_score.view(-1)
        cls_label = cls_label[inside_index]
        cls_score = cls_score[inside_index]

        # 物体的框会有重叠, 不需要NMS
        # keep_index = nms(roi, iou_maximum, self.nms_threshold)
        # keep_index = keep_index[:self.num_post_nms]
        keep_index = list(range(roi.shape[0]))

        roi = roi[keep_index]
        cls_score = cls_score[keep_index]
        cls_label = cls_label[keep_index]

        score, loc = score.reshape(-1, score.shape[-1]), loc.reshape(-1, loc.shape[-1])
        score, loc = score[inside_index][keep_index], loc[inside_index][keep_index]

        if gt_bbox is not None and gt_label is not None:
            iou = bbox_iou(torch.stack([roi[:, 1], roi[:, 0], roi[:, 3] - roi[:, 1], roi[:, 2] - roi[:, 0]], dim=1), gt_bbox)
            bbox_idx = iou.argmax(dim=1)
            iou_maximum = iou[range(iou.shape[0]), bbox_idx]

            # assign Ground Truth
            # > filter: iou > 0.5
            bg_idx = torch.where(iou_maximum < 0.6)[0]
            gt_anchor_bbox = gt_bbox[bbox_idx]
            gt_anchor_label = gt_label[bbox_idx]
            gt_anchor_label[bg_idx] = 0
            gt_anchor_loc = box2loc(anchor, gt_anchor_bbox)
            gt_anchor_loc = gt_anchor_loc[keep_index]
            gt_anchor_bbox = gt_anchor_bbox[keep_index]
            gt_anchor_label = gt_anchor_label[keep_index]


            # 此时类别极度不平衡, 上千个roi中, 99%的roi是属于背景(或者与目标物体ROI国小) => class balance
            num_pos = len(torch.where(gt_anchor_label > 0)[0])
            fg_idx = torch.where(gt_anchor_label > 0)[0].tolist()
            bg_idx = torch.where(gt_anchor_label == 0)[0].tolist()
            import random
            bg_idx = random.sample(bg_idx, num_pos) if num_pos < len(bg_idx) else bg_idx
            remain_idx = fg_idx + bg_idx
            score = score[remain_idx]
            loc = loc[remain_idx]
            roi = roi[remain_idx]
            cls_score = cls_score[remain_idx]
            gt_anchor_loc = gt_anchor_loc[remain_idx]
            gt_anchor_label = gt_anchor_label[remain_idx]
        

            return score, loc, roi, cls_score, cls_label, gt_anchor_loc, gt_anchor_label
        
        else:
            # just need fg lcass roi
            remain_idx = torch.where((cls_label > 0) & (cls_score > 0.5))[0]
            score = score[remain_idx]
            loc = loc[remain_idx]
            roi = roi[remain_idx]
            cls_score = cls_score[remain_idx]
            cls_label = cls_label[remain_idx]

            # min_size filter
            min_size_filter = torch.where(
                (roi[:, 2] > self.min_size) &
                (roi[:, 3] > self.min_size)
            )[0]
            score = score[min_size_filter]
            loc = loc[min_size_filter]
            roi = roi[min_size_filter]
            cls_score = cls_score[min_size_filter]
            cls_label = cls_label[min_size_filter]

            keep_idx = nms(
                roi,
                cls_score, 
                iou_threshold=0.8
            )[:self.num_post_nms_test]

            score = score[keep_idx]
            loc = loc[keep_idx]
            roi = roi[keep_idx]
            cls_score = cls_score[keep_idx]
            cls_label = cls_label[keep_idx]

            return score, loc, roi, cls_score, cls_label, None, None
        
