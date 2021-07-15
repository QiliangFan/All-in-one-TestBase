import torch
from torch import nn
from torch.nn.modules.dropout import Dropout
from bbox_utils.utils import bbox_iou, box2loc, loc2box
from torchvision.ops import nms
from typing import Tuple

class ROIHead(nn.Module):

    def __init__(self, in_channel: int, num_classes: int):
        super().__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Linear(in_channel, 1024),
            nn.ReLU(),
            # Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # Dropout(),
        )

        self.loc_layer = nn.Sequential(
            nn.Linear(1024, num_classes*4)
        )

        self.cls_layer = nn.Sequential(
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, rpn_out: torch.Tensor, rois: torch.Tensor, gt_anchor_label: torch.Tensor, gt_bbox: torch.Tensor, gt_label: torch.Tensor, img_size: Tuple[int, int]):
        """
        rpn_out: The output of RPN, (N_rois, last_chann * ROI_SIZE * ROI_SIZE)
        rois: (x1, y1, x2, y2)  == (N, 4)
        """
        H, W = img_size
        rois = torch.stack([rois[:, 1], rois[:, 0], rois[:, 3] - rois[:, 1], rois[:, 2] - rois[:, 0]], dim=1)

        out = self.layer1(rpn_out)
        loc = self.loc_layer(out)  # (N, num_class * 4)
        loc = loc.view(loc.shape[0], self.num_classes, 4)
        loc = loc[range(loc.shape[0]), gt_anchor_label.long()]
        loc = (loc - loc.mean()) / loc.std()
        cls = self.cls_layer(out)

        pred_box = loc2box(rois, loc)  # # (N, num_class, 4)
        pred_box_score = cls.max(dim=1)[0]

        # assign ground truth
        iou = bbox_iou(pred_box, gt_bbox)
        bbox_idx = iou.argmax(dim=1)
        iou_maximum = iou[range(iou.shape[0]), bbox_idx]
        bg_idx = torch.where(iou_maximum < 0.4)[0]
        gt_roi_bbox = gt_bbox[bbox_idx]
        gt_roi_label = gt_label[bbox_idx]
        gt_roi_label[bg_idx] = 0

        gt_roi_loc = box2loc(rois, gt_roi_bbox)

        # input : (x1, y1, x2, y2)
        keep_idx = nms(
            torch.stack([pred_box[:, 1], pred_box[:, 0], pred_box[:, 1] + pred_box[:, 3], pred_box[:, 0] + pred_box[:, 2]], dim=1)
            , iou_maximum, iou_threshold=0.5)

        pred_box = pred_box[keep_idx]
        pred_box_score = pred_box_score[keep_idx]
        rois = rois[keep_idx]
        loc = loc[keep_idx]
        cls = cls[keep_idx]
        gt_roi_label = gt_roi_label[keep_idx]
        gt_roi_loc = gt_roi_loc[keep_idx]

        return loc, cls, gt_roi_loc, gt_roi_label, pred_box, pred_box_score

    def get_inside_index(self, bbox: torch.Tensor, h: int, w: int):
        inside_index = torch.where(
            (bbox[..., 0] > 0) &
            (bbox[..., 1] > 0) &
            (bbox[..., 0] + bbox[..., 2] < h) &
            (bbox[..., 1] + bbox[..., 3] < w) 
        )[0]
        return inside_index