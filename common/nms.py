import torch
from .iou import iou
'''
    非极大值抑制非极大值抑制:删除置信度接近的锚框
    保留置信度最高的检测框，并抑制与其高度重叠（即交并比 IoU 较大）的其他检测框。
'''


# anchors: (NAC, 4)  N个边界框，每个框4个坐标:模型预测的目标锚框(相对尺寸)
# scores: (NAC,)   每个边界框的置信度得分:模型预测的目标锚框的最大类别概率
# iou_threshold: 抑制阈值
def nms(anchors, scores, iou_threshold):
    # 对预测边界框的置信度进行排序
    # anchor_inds: (N,)  按置信度降序排列的索引:原始索引
    anchor_inds = torch.argsort(scores, dim=-1, descending=True)  
    keep = []  # 保留预测边界框的原始索引

    # NMS
    while anchor_inds.numel() > 0:# 当待处理框数量大于0时，继续处理
        # 当前列表中最高置信框原始索引
        max_score_id = anchor_inds[0]
        # 保留当前最高置信框索引
        keep.append(max_score_id)
        # 如果只剩一个框，直接跳出循环
        if anchor_inds.numel() == 1: 
            break
        
        # 剩余待比较框索引
        rest_anchor_inds = anchor_inds[1:]

        # 取出当前最高置信框
        # max_score_anchor: (1, 4)
        max_score_anchor = anchors[max_score_id, :].reshape(-1, 4)
        # 取出剩余待比较框
        # rest_anchors: (N-1, 4)
        rest_anchors = anchors[rest_anchor_inds, :].reshape(-1, 4)

        # anchor_iou: (1,N-1)->(N-1,)
        # 此时anchor_iou并不是从大到小顺序序列
        anchor_iou = iou(max_score_anchor, rest_anchors).reshape(-1)
        # 筛选出 IoU 小于阈值的框自然索引,保留，删除太相似的anchor
        # mask(R,)
        mask = anchor_iou <= iou_threshold
        # 跳过第一个框，因为它是当前最高置信框
        # 更新anchor索引：去掉本轮最大置信框和相似框，保留其他框，用于下一轮计算
        # anchor_inds: (R,)
        anchor_inds = anchor_inds[1:][mask]
    # keep(k,): 最终保留的框索引(原始索引)
    keep = torch.tensor(keep, device=anchors.device)
    return keep
