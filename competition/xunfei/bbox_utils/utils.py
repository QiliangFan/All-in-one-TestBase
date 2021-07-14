from typing import Sequence, Tuple
import torch
from torch._C import device


def loc2box(anchor: torch.Tensor, loc: torch.Tensor):
    """
    anchor: (H, W, n_anchor, 4)
    loc: (H, W, n_anchor, 4)
    """
    roi = torch.zeros_like(anchor, device=loc.device)

    ctr_anchor_y = anchor[:, :, :, 0] + 0.5 * anchor[:, :, :, 2]
    ctr_anchor_x = anchor[:, :, :, 1] + 0.5 * anchor[:, :, :, 3]
    anchor_height = anchor[:, :, :, 2]
    anchor_width = anchor[:, :, :, 3]

    loc_y = loc[:, :, :, 0]
    loc_x = loc[:, :, :, 1]
    loc_height = loc[:, :, :, 2]
    loc_width = loc[:, :, :,3]

    ctr_roi_y = ctr_anchor_y + loc_y * anchor_height
    ctr_roi_x = ctr_anchor_x + loc_x * anchor_width
    roi_height = anchor_height * torch.exp(loc_height)
    roi_width = anchor_width * torch.exp(loc_width)

    roi[:, :, :, 0] = ctr_roi_y - 0.5 * roi_height
    roi[:, :, :, 1] = ctr_roi_x - 0.5 * roi_width
    roi[:, :, :, 2] = roi_height
    roi[:, :, :, 3] = roi_width

    return roi


def box2loc(anchor: torch.Tensor, gt_bbox: torch.Tensor):
    """
    anchor: (N, 4)
    gt_bbox: (N, 4)
    """
    loc = torch.zeros_like(anchor, device=gt_bbox.device)

    ctr_anchor_y = anchor[:, 0] + 0.5 * anchor[:, 2]
    ctr_anchor_x = anchor[:, 1] + 0.5 * anchor[:, 3]
    anchor_height = anchor[:, 2]
    anchor_width = anchor[:, 3]

    ctr_bbox_y = gt_bbox[:, 0] + 0.5 * gt_bbox[:, 2]
    ctr_bbox_x = gt_bbox[:, 1] + 0.5 * gt_bbox[:, 3]
    bbox_height = gt_bbox[:, 2]
    bbox_width = gt_bbox[:, 3]

    loc_y = (ctr_bbox_y - ctr_anchor_y) / anchor_height
    loc_x = (ctr_bbox_x - ctr_anchor_x) / anchor_width
    loc_height = torch.log(bbox_height / anchor_height)
    loc_width = torch.log(bbox_width / anchor_width)

    loc[:, 0] = loc_y
    loc[:, 1] = loc_x
    loc[:, 2] = loc_height
    loc[:, 3] = loc_width

    return loc

def get_inside_index(box: torch.Tensor, h: int, w: int):
    """
    box: (x1, y1, x2, y2)
    """
    tolerance = (h + w) / 20
    inside_index = torch.where(
        (box[:, 0] >= -tolerance) &
        (box[:, 1] >= -tolerance) &
        (box[:, 2] <= w + tolerance) &
        (box[:, 3] <= h + tolerance)
    )[0]
    return inside_index


def bbox_iou(src: torch.Tensor, dst: torch.Tensor):
    """
    (y, x, h, w)
    src: (N_src, 4)
    dst: (N_dst, 4)
    """
    assert src.shape[1] == 4 and dst.shape[1] == 4

    max_left_top = torch.maximum(src[:, None, :2], dst[:, :2])
    min_right_bottom = torch.minimum(src[:, None, :2] + src[:, None, 2:], dst[:, :2] + dst[:, 2:])

    inter_area = torch.prod(min_right_bottom - max_left_top, dim=2) * (max_left_top < min_right_bottom).all(dim=2)
    area_a = torch.prod(src[:, 2:], dim=1)
    area_b = torch.prod(dst[:, 2:], dim=1)
    union_area = area_a[:, None] + area_b - inter_area
    return inter_area / union_area