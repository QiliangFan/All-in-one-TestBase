import torch

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    cy, cx = base_size / 2, base_size / 2

    anchor_base = torch.zeros((len(ratios) * len(anchor_scales), 4), dtype=torch.float32)
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * torch.sqrt(ratios[i])  # 这里不加sqrt也行
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

