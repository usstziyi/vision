import torch
from torch.nn import functional as F
from .iou import box_iou
from .anchor_shift import anchor_shift
'''
    非极大值抑制非极大值抑制:删除置信度接近的锚框
    保留置信度最高的检测框，并抑制与其高度重叠（即交并比 IoU 较大）的其他检测框。
'''


# 非极大值抑制(1张图中所有特征图中所有预测框的NMS处理，非批量)
# pred_boxes: (P*A,4)
# pred_score: (P*A,) :每个预测框的最大前景概率
# iou_threshold: 抑制阈值
# 混合类别的 NMS
def nms(pred_boxes, pred_score, iou_threshold):
    # 对预测边界框的置信度进行排序
    # sorted_boxes(P*A,)  按置信度降序排列的索引:从高到低
    # 保留了原始序列
    sorted_boxes = torch.argsort(pred_score, descending=True)  
    keep_indices = []  # 保存原始序列

    # NMS
    # 挨个处理预测框
    while sorted_boxes.numel() > 0:
        max_indices = sorted_boxes[0]   # 最大置信度的索引
        rest_indices = sorted_boxes[1:] # 剩余索引
        # 保留当前最高置信框索引
        keep_indices.append(max_indices)
        # 如果只剩一个框，直接跳出循环
        if sorted_boxes.numel() == 1: 
            break

        # head_box(1,4)
        head_box = pred_boxes[max_indices, :].reshape(-1, 4)
        # rest_boxes(P*A-1,4)
        rest_boxes = pred_boxes[rest_indices, :].reshape(-1, 4)
        # 计算两两 IoU
        # iou(P*A-1,)
        iou = box_iou(head_box, rest_boxes).reshape(-1)
        # 筛选出 IoU 小于阈值的框自然索引
        # survive_indices(S,)
        survive_indices = torch.nonzero(iou <= iou_threshold).reshape(-1)
        # 转换为总绝对索引
        survive_indices = survive_indices + 1
        # 下一轮参与NMS的框索引
        sorted_boxes = sorted_boxes[survive_indices]
    
    # keep_indices(K,):从P*A中筛选出最终保留的框索引
    keep_indices = torch.tensor(keep_indices, device=pred_boxes.device)
    # keep_indices(K,) :保留的原始序列索引
    return keep_indices


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




"""
    使用非极大值抑制来预测质量的边界框
    假设我们在检测图像中的猫：
    - 图像中有一只猫(类别1)
    - 生成了1000个锚框
    - 只有10个锚框与猫有较好重叠(标记为类别1)
    - 其余990个锚框与猫无重叠(标记为类别0,即背景)
    - 在NMS后,一些低置信度的预测被标记为-1(忽略):这些锚框不是背景，是被丢弃的低质量前景
"""
# 预测阶段用(可批量)
# batch_anchors(1,P*A,4)
# batch_pred_classes(B,P*A*C)
# batch_pred_offset(B,P*A*4)
def filter_boxes_by_nms(batch_anchors, batch_pred_classes, batch_pred_offset, num_classes):    
    nms_threshold=0.5         # 重叠框,置为背景
    pos_threshold=0.009999999 # 得分低的框，置位背景
    device = batch_pred_classes.device

    # anchors(1, NAC, 4) -> anchors(NAC, 4)
    anchors = batch_anchors.squeeze(0)
    # (B,P*A,C)
    batch_pred_classes = batch_pred_classes.reshape(batch_pred_classes.shape[0],-1,num_classes)
    batch_pred_offset = batch_pred_offset.reshape(batch_pred_offset.shape[0],-1,4)


    # softmax
    # 从“分数”到“概率”
    # (B,P*A,C) 
    batch_pred_classes = F.softmax(batch_pred_classes, dim=-1)

    
    batch_size = batch_pred_classes.shape[0]
    num_anchors = batch_pred_classes.shape[1]
    num_classes = batch_pred_classes.shape[2]




    list_box_info = []
    for i in range(batch_size):
        pred_classes = batch_pred_classes[i] # (P*A,C)
        pred_offset = batch_pred_offset[i] # (P*A,4)

        # -1 0 1 2 3 4   pred_score class_id
        # * * * * * *  -> *         2
        # * * * * * *  -> *         1
        # * * * * * *  -> *         3
        # * * * * * *  -> *         0
        # * * * * * *  -> *         4
        # * * * * * *  -> *         3
        # * * * * * *  -> *         4
        # * * * * * *  -> *         2
        # * * * * * *  -> *         3
        # * * * * * *  -> *         4
        # 先假设所有框都是某种物体 -> 算出物体的置信度 -> 用 NMS 去除重叠的物体框 -> 最后把置信度太低的框（其实是背景）扔掉。
        # pred_score(P*A,)
        # class_id(P*A,)
        # 得到原始序列
        pred_score, class_id = torch.max(pred_classes[:,1:], dim=-1)

        # 移动锚框到预测框
        # anchors(P*A,4)(xa, ya, wa, ha)
        # offsets(P*A,4)(dx, dy, dw, dh) (已标准化)
        # pred_boxes(P*A,4)(xb, yb, wb, hb)
        pred_boxes = anchor_shift(anchors, pred_offset)
        # 非极大值抑制
        # 混合类别的 NMS:先定类别，后做NMS,不会出现同一个预测框被分配为两个类别的情况
        # 如果两个重叠很高的框，一个是“猫（0.9）”，一个是“狗（0.8）”。
        # NMS 会保留分数高的“猫”，抑制掉“狗”。
        # keep_indices(K,) :保留的原始序列索引
        keep_indices = nms(pred_boxes, pred_score, nms_threshold)

        # 找到所有的non_keep索引，并将类设置为背景(现在是预测阶段)
        all_indices = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep_indices, all_indices))
        # 比如 [0, 2, 0, 1, 2, 3] 
        # uniques ：所有唯一元素 [0, 1, 2, 3] 
        # counts ：每个唯一元素在 combined 中出现的次数 [2, 1, 2, 1] 
        uniques, counts = combined.unique(return_counts=True)
        # non_keep(P*A-K,)
        non_keep = uniques[counts == 1]
        # all_indices(P*A,) 原始序列索引
        all_indices = torch.cat((keep_indices, non_keep), dim=0)
        # 把重叠的框设置为背景-1
        class_id[non_keep] = -1

        # 从原始序列中取出对应的置信度和边界框坐标
        pred_score = pred_score[all_indices]
        pred_boxes = pred_boxes[all_indices]
        class_id = class_id[all_indices]

        # 把低质量的框设置为背景-1
        below_min_indices = (pred_score < pos_threshold)
        class_id[below_min_indices] = -1
        pred_score[below_min_indices] = 1 - pred_score[below_min_indices]
        
        
        # pred_boxes(P*A, 4)
        # pred_score(P*A,1)
        # class_id(P*A,1)
        # box_info(P*A, 6)
        box_info = torch.cat((pred_boxes,pred_score.unsqueeze(1),class_id.unsqueeze(1)), dim=1)
        list_box_info.append(box_info)
        # batch_box_info(B, P*A, 6)
        batch_box_info = torch.stack(list_box_info, dim=0)
    return batch_box_info
