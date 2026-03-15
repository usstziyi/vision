import torch


def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height).

    Defined in :numref:`sec_bbox`"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), dim=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right).

    Defined in :numref:`sec_bbox`"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), dim=-1)
    return boxes

# 把偏移量施加到锚框上，得到预测框
# 注：这个函数是预测阶段用的，不是训练阶段用的
# offsets是预测的偏移量，因为预测阶段没有标签可用
# 这个函数返回锚框+预测后的结果
# anchors(P*A,4)(xa, ya, wa, ha)
# offsets(P*A,4)(dx, dy, dw, dh) (已标准化)
def anchor_shift(anchors, offsets):
    # 确保输入是浮点数以防整数除法问题
    anchors = anchors.float()
    offsets = offsets.float()  

    # 转换为中心点表示
    anchors = box_corner_to_center(anchors)



    # 1. 逆向标准化 (Denormalization)
    # 原逻辑: norm_val = raw_val / std
    # 逆逻辑: raw_val = norm_val * std
    stds = torch.tensor([0.1, 0.1, 0.2, 0.2], device=anchors.device, dtype=anchors.dtype)
    
    # 提取并还原原始偏移量
    dx = offsets[:, 0] * stds[0]
    dy = offsets[:, 1] * stds[1]
    dw = offsets[:, 2] * stds[2]
    dh = offsets[:, 3] * stds[3]

    # 提取锚框参数
    xa = anchors[:, 0]
    ya = anchors[:, 1]
    wa = anchors[:, 2]
    ha = anchors[:, 3]

    # 2. 逆向计算真实框坐标
    xb = xa + dx * wa
    yb = ya + dy * ha
    

    # 注意：原函数中有 eps = 1e-6 在 log 内。
    # 严格数学逆运算是 wa * (exp(dw) - eps)，但在深度学习实践中，
    # 由于 eps 极小且 exp(dw) 通常远大于 eps，通常直接写作 wa * exp(dw)。
    # 这里为了数值稳定且符合常规检测头解码逻辑，使用 exp。
    wb = wa * torch.exp(dw)
    hb = ha * torch.exp(dh)

    # pred_boxes (P*A, 4) -> (xb, yb, wb, hb)
    pred_boxes = torch.stack([xb, yb, wb, hb], dim=1)

    # 转换为角点表示
    pred_boxes = box_center_to_corner(pred_boxes)

    
    return pred_boxes


def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets.

    Defined in :numref:`subsec_labeling-anchor-boxes`"""
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox