from utils.bbox_tools import loc2box
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from typing import Sequence, Tuple


class FasterRCNN(nn.Module):

    def __init__(self, extractor: nn.Module, rpn: nn.Module, head: nn.Module, loc_normalize_mean=(0., 0., 0., 0.), loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        super().__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset("evaluate")

    def forward(self, x: torch.Tensor, scale=1.) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """[summary]

        Args:
            x (torch.Tensor): [description]
            scale (float, optional): 这个scale是读取图片后预处理时缩放的scale, 和后面的stride要区分开. Defaults to 1..
        """
        img_size = x.shape[2:]  # RGB 的通道可以放在channel处, PNG的channel则为1

        h = self.extractor(x)
        rpn_locs, rpn_scores, rois, roi_indices, anchor, rpn_debug_score = self.rpn(
            h, img_size)
        if isinstance(roi_indices, list):  # since batch size = 1
            roi_indices = roi_indices[0]
        if isinstance(rois, list):
            rois = rois[0]
        roic_cls_locs, roi_scores = self.head(h, rois, roi_indices)
        return roic_cls_locs, roi_scores, rois, roi_indices
    
    @torch.no_grad()
    def predict(self, imgs: Sequence[torch.Tensor], sizes=None, visualize=False):
        """[summary]

        Args:
            img (torch.Tensor): iterable of Tensor. All images are in CHW and RGB format
            sizes ([type], optional): 图像的原本尺寸. Defaults to None.
            visualize (bool, optional): [description]. Defaults to False.
        """
        self.eval()
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(imgs, sizes):
            img = img[None].float()  # expand with batch size 1
            scale = img.shape[-1] / size[-1]
            roi_cls_locs, roi_scores, rois, _ = self(img, scale=scale)
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_locs.data
            roi: torch.Tensor = rois / scale  # 将roi还原到原图

            mean = torch.as_tensor(self.loc_normalize_mean).repeat(self.n_class)[None].to(device=roi_cls_loc.device)
            std = torch.as_tensor(self.loc_normalize_std).repeat(self.n_class)[None].to(device=roi_cls_loc.device)

            roi_cls_loc: torch.Tensor = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.contiguous().view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2box(roi.reshape((-1, 4)), roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox.contiguous().view(-1, self.n_class*4)

            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0]-1)
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1]-1)

            prob = F.softmax(roi_score, dim=1)

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        
        self.use_preset("evaluate")
        self.train()
        return bboxes, labels, scores

    @property
    def n_class(self) -> int:
        """Total number of classes including the background

        Returns:
            int: the number of classes
        """
        return self.head.n_class

    def use_preset(self, preset: str):
        if preset == "visualize":
            self.nms_thresh = 0.3
            self.score_thresh = 0.7  # discard bbox with too low confidence
        elif preset == "evaluate":
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError("preset must be visualizer or evaluate")

    def _suppress(self, raw_cls_bbox: torch.Tensor, raw_prob: torch.Tensor):
        bbox = list()
        label = list()
        score = list()

        for i in range(1, self.n_class):
            cls_bbox_i = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, i, :]
            prob_i = raw_prob[:, i]
            mask = prob_i > self.score_thresh
            cls_bbox_i = cls_bbox_i[mask]
            prob_i = prob_i[mask]
            keep = nms(cls_bbox_i[:, [1, 0, 3, 2]], prob_i, self.nms_thresh)
            bbox.append(cls_bbox_i[keep])
            label.append((i-1) * torch.ones((len(keep), ), device=raw_cls_bbox.device))
            score.append(prob_i[keep])
        bbox = torch.cat(bbox, dim=0).type(torch.float32)
        label = torch.cat(label, dim=0).type(torch.int32)
        score = torch.cat(score, dim=0).type(torch.float32)
        return bbox, label, score

