from torchvision.models import vgg16
import torch
from torch import nn
from faster_rcnn import FasterRCNN
from rpn import RPN
from torchvision.ops import RoIPool

def decom_vgg16():
    model = vgg16()

    features = list(model.features)[:30]
    classifier = model.classifier
    classifier = list(classifier)

    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier


class FasterRCNNVGG(FasterRCNN):
    feat_stride = 16

    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]):
        extractor, classifier = decom_vgg16()

        rpn = RPN(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_size=self.feat_stride
        )

        head = VGGROIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super().__init__(extractor, rpn, head)

        
class VGGROIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier: nn.Module):
        """[summary]

        Args:
            n_class ([type]): [description]
            roi_size ([type]): [description]
            spatial_scale ([type]): input image 到 feature_map的缩放尺寸 1 / feat_stride
            classifier (nn.Module): [description]
        """
        super(VGGROIHead, self).__init__()

        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi = RoIPool((self.roi_size, self.roi_size), self.spatial_scale)

    def forward(self, x: torch.Tensor, rois: torch.Tensor, roi_indices: torch.Tensor):
        roi_indices = roi_indices.type(torch.float32)
        rois = rois.type(torch.float32)
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]  # yx -> xy
        indices_and_rois = xy_indices_and_rois.contiguous()

        pool: torch.Tensor = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)
        fc7 = self.classifier(pool)
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores
