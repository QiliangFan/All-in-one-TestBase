from utils.bbox_tools import loc2box
import torch
import torch.nn as nn
from torchvision.ops import nms

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
        keep = torch.where((hs > min_size) & (ws > min_size))
        roi, score = roi[keep, :], score[keep]

        order = score.ravel().argsort()[::-1]  # largs -> small
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi, score = roi[order, :], score[order]

        keep = nms(roi, score, self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi