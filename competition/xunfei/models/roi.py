import torch
from torch import nn
from bbox_utils.utils import bbox_iou, box2loc, loc2box
from torchvision.ops import nms


class ROIHead(nn.Module):

    def __init__(self, in_channel: int, num_classes: int):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.Linear(in_channel, in_channel),
            nn.Linear(in_channel, in_channel),
        )

        self.loc_layer = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.Linear(in_channel, 4)
        )

        self.cls_layer = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.Linear(in_channel, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, rpn_out: torch.Tensor, rois: torch.Tensor,  gt_bbox: torch.Tensor, gt_label: torch.Tensor):
        """
        rpn_out: The output of RPN, (N_rois, last_chann * ROI_SIZE * ROI_SIZE)
        rois: (x1, y1, x2, y2)  == (N, 4)
        """
        rois = torch.stack([rois[:, 1], rois[:, 0], rois[:, 3] - rois[:, 1], rois[:, 2] - rois[:, 0]], dim=1)

        out = self.layer1(rpn_out)
        loc = self.loc_layer(out)
        cls = self.cls_layer(out)

        pred_box = loc2box(rois, loc)
        pred_box_score = cls.max(dim=1)[0]

        # input : (x1, y1, x2, y2)
        keep_idx = nms(
            torch.stack([pred_box[:, 1], pred_box[:, 0], pred_box[:, 1] + pred_box[:, 3], pred_box[:, 0] + pred_box[:, 2]], dim=1)
            , pred_box_score, iou_threshold=0.7)

        pred_box = pred_box[keep_idx]
        pred_box_score = pred_box[keep_idx]
        rois = rois[keep_idx]
        loc = loc[keep_idx]
        cls = cls[keep_idx]

        # assign ground truth
        iou = bbox_iou(pred_box, gt_bbox)
        bbox_idx = iou.argmax(dim=1)
        iou_maximum = iou[range(iou.shape[0]), bbox_idx]
        bg_idx = torch.where(iou_maximum < 0.8)[0]
        gt_roi_bbox = gt_bbox[bbox_idx]
        gt_roi_label = gt_label[bbox_idx]
        gt_roi_label[bg_idx] = 0

        gt_roi_loc = box2loc(rois, gt_roi_bbox)

        return loc, cls, gt_roi_loc, gt_roi_label, pred_box