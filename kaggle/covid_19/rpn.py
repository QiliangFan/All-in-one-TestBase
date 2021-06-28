from torch.nn.modules.batchnorm import BatchNorm2d
from utils.creator_tool import ProposalCreator
from torch.nn import functional as F
import torch
from torch import nn
from utils.bbox_tools import generate_anchor_base
from resnet import BasicBlock


class RPN(nn.Module):

    def __init__(self,in_channel, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_size=16, proposal_creator_params=dict()
                 ):
        super().__init__()
        self.anchor_base = generate_anchor_base(
            ratios=ratios, anchor_scales=anchor_scales)
        self.feat_size = feat_size
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)
        n_anchor = self.anchor_base.shape[0]  # 每个像素多少个anchor
        self.conv1 = nn.Sequential(
            BasicBlock(in_channel, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels)
        )
        self.score = BasicBlock(mid_channels, n_anchor * 2)
        self.loc = BasicBlock(mid_channels, n_anchor * 4)

    def forward(self, x: torch.Tensor, img_size):
        n, _, hh, ww = x.shape
        self.anchor_base = self.anchor_base.to(x.device)
        anchor = _enumerate_shifted_anchor(self.anchor_base, self.feat_size, hh, ww)

        n_anchor = anchor.shape[0] // (hh * ww)  # 每个像素多少个anchor, 是特征图上的, 由于尺寸变小了个数会增多
        h = F.relu(self.conv1(x), inplace=True)

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
        rpn_debug_score = list()
        for i in range(n):
            roi, score = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size)
            batch_index = i * torch.ones((len(roi),), dtype=torch.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
            rpn_debug_score.append(score)
        return rpn_locs, rpn_scores, rois, roi_indices, anchor, rpn_debug_score


def _enumerate_shifted_anchor(anchor_base: torch.Tensor, feat_stride, height, width):
    shift_y = torch.arange(0, height * feat_stride, feat_stride)
    shift_x = torch.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = torch.meshgrid(shift_x, shift_y)
    shift = torch.stack((shift_y.flatten(), shift_x.flatten(), shift_y.flatten(), shift_x.flatten()), dim=1)
    shift = shift.to(anchor_base.device)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + shift.reshape((1, K, 4)).permute((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).type(torch.float32)
    return anchor