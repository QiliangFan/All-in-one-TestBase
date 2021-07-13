from typing import Tuple
import torch
from torch import nn
from bbox_utils.proposal import ProposalCreator
from torchvision.ops import roi_pool

class RPN(nn.Module):
    roi_pool_size = 7

    def __init__(self, in_channel: int, num_classes: int, n_anchor: int):
        super().__init__()

        self.num_classes = num_classes
        self.n_anchor = n_anchor

        self.proposal_creator = ProposalCreator()

        self.cls_layer = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            ) for i in range(5)],
            nn.Conv2d(in_channel, n_anchor * num_classes, kernel_size=3, stride=1, padding=1)
        )

        self.loc_layer = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1),
            ) for i in range(5)],
            nn.Conv2d(in_channel, n_anchor * 4, kernel_size=3, stride=1, padding=1)
        )

    def forward(
            self, 
            features: torch.Tensor, 
            anchors: torch.Tensor,
            feat_stride: int,
            img_size: Tuple[int, int]) -> Tuple[torch.Tensor, ...]:
        n_batch = features.shape[0]
        assert n_batch == 1
        H, W = features.shape[2:4]

        cls: torch.Tensor = self.cls_layer(features) # (N, C, H, W)

        loc: torch.Tensor = self.loc_layer(features)

        cls = cls.permute(dims=[0, 2, 3, 1]).view(n_batch, H, W, self.n_anchor, self.num_classes)
        cls_softmax = torch.softmax(cls, dim=-1).to(device=cls.device)
        loc = loc.permute(dims=[0, 2, 3, 1]).view(n_batch, H, W, self.n_anchor, 4)

        roi, roi_score, roi_label = self.proposal_creator(cls_softmax, loc, anchors, img_size)
        _roi = torch.cat([torch.zeros((roi.shape[0], 1), device=features.device), roi], dim=1)
        out = roi_pool(features, _roi, output_size=self.roi_pool_size, spatial_scale=1./feat_stride)
        out = out.view(out.shape[0], -1)
        return out, cls_softmax, loc, roi, roi_score, roi_label

