import torch
from .iou import iou
'''
    非极大值抑制非极大值抑制:删除置信度接近的锚框
'''


# anchors: (NAC, 4)  N个边界框，每个框4个坐标
# scores: (NAC,)   每个边界框的置信度得分
# iou_threshold: 抑制阈值
def nms(anchors, scores, iou_threshold):
    # 对预测边界框的置信度进行排序
    # anchor_inds: (N,)  按置信度降序排列的索引:原始索引
    anchor_inds = torch.argsort(scores, dim=-1, descending=True)  
    keep = []  # 保留预测边界框的指标

    # NMS
    while anchor_inds.numel() > 0:# 当待处理框数量大于0时，继续处理
        # 当前最高置信框原始索引
        max_score_id = anchor_inds[0]
        # 剩余待比较框索引
        rest_anchor_inds = anchor_inds[1:]
        # 保留当前最高置信框索引
        keep.append(max_score_id)
        # 如果只剩一个框，直接跳出循环
        if anchor_inds.numel() == 1: 
            break
        
        # 取出当前最高置信框
        # max_score_anchor: (1, 4)
        max_score_anchor = anchors[max_score_id, :].reshape(-1, 4)
        # 取出剩余待比较框
        # rest_anchors: (N-1, 4)
        rest_anchors = anchors[rest_anchor_inds, :].reshape(-1, 4)

        # anchor_iou: (1,N-1)->(N-1,)
        anchor_iou = iou(max_score_anchor, rest_anchors).reshape(-1)
        # 筛选出 IoU 小于阈值的框自然索引,保留，删除太相似的anchor
        # inds: (M,)  保留的索引，M <= N-1
        # torch.nonzero返回的是满足条件的自然，从0开始
        inds = torch.nonzero(anchor_iou <= iou_threshold).reshape(-1)
        # 转换为总绝对索引
        inds = inds + 1
        # 下一轮参与NMS的框索引(一直沿用原始索引)
        anchor_inds = rest_anchor_inds[inds]
    # keep(k,): 最终保留的框索引(原始索引)
    keep = torch.tensor(keep, device=anchors.device)
    return keep
