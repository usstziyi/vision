import torch
from .generate import generate_anchors

def generate_fmap_anchors(fmap_w, fmap_h, scales, ratios=[1, 2, 0.5]):
    # 前两个维度上的值不影响输出
    # 等效特征图(B, C, h, w)
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # 在特征图上生成锚框
    anchors = generate_anchors(fmap, scales, ratios)
    return anchors