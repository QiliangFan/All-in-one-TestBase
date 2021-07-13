from typing import Sequence, Tuple
import torch


def loc2box(anchor: torch.Tensor, loc: torch.Tensor):
    roi = torch.zeros_like(anchor, device=loc.device)

    roi[:, :, :, 0] = anchor[:, :, :, 0] + loc[:, :, :, 0] * anchor[:, :, :, 2]
    roi[:, :, :, 1] = anchor[:, :, :, 1] + loc[:, :, :, 1] * anchor[:, :, :, 3]
    roi[:, :, :, 2] = anchor[:, :, :, 2] * torch.exp(loc[:, :, :, 2])
    roi[:, :, :, 3] = anchor[:, :, :, 3] * torch.exp(loc[:, :, :, 3])

    return roi



def get_inside_index(box: torch.Tensor, h: int, w: int):
    """
    box: (x1, y1, x2, y2)
    """
    inside_index = torch.where(
        (box[:, 0] > 0) &
        (box[:, 1] > 0) &
        (box[:, 2] < h) &
        (box[:, 3] < w)
    )[0]
    return inside_index


