from utils.creator_tool import ProposalCreator
from torch.nn import functional as F
import torch
from torch import nn
from utils.bbox_tools import generate_anchor_base


class RPN(nn.Module):

    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_size=16, proposal_creator_params=dict()
                 ):
        super().__init__()
        self.anchor_base = generate_anchor_base(
            ratios=ratios, anchor_scales=anchor_scales)
        self.feat_size = feat_size
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]  # 每个像素多少个anchor
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        self.score = nn.Conv2d(mid_channels, n_anchor*2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channels, n_anchor*4, 1, 1, 0)

    def forward(self, x: torch.Tensor, img_size, scale=1.):
        n, _, hh, ww = x.shape
        anchor = _enumerate_shifted_anchor(self.anchor_base, self.feat_size, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)  # 每个像素多少个anchor, 是特征图上的, 由于尺寸变小了个数会增多
        h = F.relu(self.conv1(x))

        rpn_locs: torch.Tensor = self.loc(h)
        rpn_scores: torch.Tensor = self.score(h)

        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)  # 按照空间位置\anchor循序, 排列输出的locs
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4)  # 只对每个anchor的正负例进行softmax
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()  # 只要物体的分数
        rpn_fg_scores = rpn_fg_scores.view(n, -1)
        rpn_scores = rpn_scores.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale=scale)
            batch_index = i * torch.ones((len(roi),), dtype=torch.int32)
            rois.append(roi)
            roi_indices.append(roi_indices)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

    

def _enumerate_shifted_anchor(anchor_base: torch.Tensor, feat_stride, height, width):
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
    shift = torch.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), dim=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).permute((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).type(torch.float32)
    return anchor