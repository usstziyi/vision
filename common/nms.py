import torch
from .iou import box_iou
'''
    非极大值抑制非极大值抑制:删除置信度接近的锚框
    保留置信度最高的检测框，并抑制与其高度重叠（即交并比 IoU 较大）的其他检测框。
'''


# 非极大值抑制
# boxes: (N, 4)  N个边界框，每个框4个坐标
# scores: (N,)   每个边界框的置信度得分
# iou_threshold: 抑制阈值
def nms(boxes, scores, iou_threshold):
    # 对预测边界框的置信度进行排序
    # B: (N,)  按置信度降序排列的索引:从高到低
    sorted_score = torch.argsort(scores, dim=-1, descending=True)  
    keep_indices = []  # 保留预测边界框的指标

    # NMS
    while sorted_score.numel() > 0:# 当待处理框数量大于0时，继续处理
        # 本轮最高置信框索引
        max_indices = sorted_score[0]
        rest_indices = sorted_score[1:]
        # 保留当前最高置信框索引
        keep_indices.append(max_indices)
        # 如果只剩一个框，直接跳出循环
        if sorted_score.numel() == 1: 
            break


        
        # 取出当前最高置信框
        current_box = boxes[max_indices, :].reshape(-1, 4)          # (1, 4)
        # 取出剩余待比较框
        rest_boxes = boxes[rest_indices, :].reshape(-1, 4)       # (N-1, 4)
        # 计算两两 IoU(1,N-1)->(N-1,)
        # (0,1,2,3,4)
        # (0-1,0-2,0-3,0-4)
        iou = box_iou(current_box, rest_boxes).reshape(-1)  # (N-1,)
        # 筛选出 IoU 小于阈值的框自然索引
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)  # inds: (M,)  保留的索引，M <= N-1
        # 转换为总绝对索引
        inds = inds + 1
        # 下一轮参与NMS的框索引
        sorted_score = sorted_score[inds]

    return torch.tensor(keep_indices, device=boxes.device)  # (K,)  K为最终保留的框数量


# pytorch自带的nms函数
# from torchvision.ops import nms
# import torch

# # 假设 boxes 和 scores 已经定义
# # boxes: (N, 4) 格式为 [x1, y1, x2, y2]
# # scores: (N,)
# # iou_threshold: 标量

# keep_indices = nms(boxes, scores, iou_threshold)

# # keep_indices 是一个包含保留框索引的 Tensor (K,)
# # 你可以通过它来获取最终的框和得分：
# final_boxes = boxes[keep_indices]
# final_scores = scores[keep_indices]