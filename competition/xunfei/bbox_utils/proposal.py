from typing import Tuple
from six import class_types
import torch
import numpy as np
from .utils import loc2box, get_inside_index
from torchvision.ops import nms


class ProposalCreator:

    def __init__(self, num_post_nms: int = 3000, nms_threshold: float = 0.7):
        """[summary]

        Args:
            num_post_nms (int): The number of remained samples after NMS
            nms_threshold (float): nms_threshold
        """
        self.num_post_nms = num_post_nms
        self.nms_threshold = nms_threshold

    def __call__(self, score: torch.Tensor, loc: torch.Tensor, anchor: torch.Tensor, img_size: Tuple[int, int]):
        """[summary]

        Args:
            score (torch.Tensor): [n_batch, H, W, n_anchor, num_class]
            loc (torch.Tensor): [n_batch, H, W, n_anchor, 4]
            anchor (torch.Tensor): [H, W, n_anchor, 4]
        """
        n_batch = score.shape[0]
        assert n_batch == 1

        H, W = img_size

        score = score[0]
        loc = loc[0]

        # roi
        roi = loc2box(anchor, loc)
        roi = roi.view(-1, 4).contiguous()
        roi[:, 2] = roi[:, 0] + roi[:, 2]
        roi[:, 3] = roi[:, 0] + roi[:, 3]
        roi = roi[:, [1, 0, 3, 2]]  # (x1, y1, x2, y2)
        inside_index = get_inside_index(roi, H, W)
        roi = roi[inside_index]

        # cls & NMS
        cls_score, cls_label = score.max(dim=-1)  # [H, W, n_anchor]
        cls_label = cls_label.view(-1)
        cls_score = cls_score.view(-1)
        cls_label = cls_label[inside_index]
        cls_score = cls_score[inside_index]

        keep_index = nms(roi, cls_score, self.nms_threshold)
        keep_index = keep_index[:self.num_post_nms]

        roi = roi[keep_index]
        cls_score = cls_score[keep_index]
        cls_label = cls_label[keep_index]

        return roi, cls_score, cls_label
        
