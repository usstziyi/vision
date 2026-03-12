import torch

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
    
    return pred_boxes
