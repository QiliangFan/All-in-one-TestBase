import torch


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    cy, cx = base_size / 2, base_size / 2

    anchor_base = torch.zeros(
        (len(ratios) * len(anchor_scales), 4), dtype=torch.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * \
                torch.sqrt(ratios[i])  # 这里不加sqrt也行
            w = base_size * anchor_scales[j] * torch.sqrt(1 / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = cy - h / 2
            anchor_base[index, 1] = cx - w / 2
            anchor_base[index, 2] = cy + h / 2
            anchor_base[index, 3] = cx + w / 2
    return anchor_base


def loc2box(src_bbox: torch.Tensor, loc: torch.Tensor):
    if src_bbox.shape[0] == 0:
        return torch.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.type_as(loc)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_weight = src_bbox[:, 3] - src_bbox[:, 1]
    src_c_y = src_bbox[:, 0] + 0.5 * src_weight
    src_c_x = src_bbox[:, 1] + 0.5 * src_height

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, None] + src_c_y[:, None]
    ctr_x = dx * src_weight[:, None] + src_c_x[:, None]
    h = torch.exp(dh) * src_height[:, None]
    w = torch.exp(dw) * src_weight[:, None]

    dst_bbox = torch.zeros_like(loc, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox


def bbox2loc(src_bbox: torch.Tensor, dst_bbox: torch.Tensor):
    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = torch.log(base_height / height)
    dw = torch.log(base_width / width)

    loc = torch.stack([dy, dx, dh, dw], dim=1)
    return loc


def bbox_iou(bbox_a: torch.Tensor, bbox_b: torch.Tensor):
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError()

    # 最大的左上角与最大的最小的右下角, 就构成IOU面积
    top_left = torch.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    bottom_right = torch.maximum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_inter = torch.prod(bottom_right - top_left, dim=2) * \
        (top_left < bottom_right).all(dim=2)
    area_a = torch.prod(bbox_a[:, 2:] - bbox_b[:, :2], dim=1)
    area_b = torch.prod(bbox_b[:, 2:] - bbox_b[:, :2], dim=1)
    return area_inter / (area_a[:, None] + area_b - area_inter)
