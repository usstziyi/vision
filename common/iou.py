import torch

"""
    计算两组锚框的交并比
"""

# shape(box数量，4):[xmin, ymin, xmax, ymax]
def box_iou(boxes1, boxes2):
 
    # 定一个内部函数：计算边界框的面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    # shape(box数量，)
    areas1 = box_area(boxes1)
    # shape(box数量，)
    areas2 = box_area(boxes2)


    # None 在 PyTorch/NumPy 索引中等同于 unsqueeze 操作，它在指定位置插入一个大小为 1 的新维度。
    # (N,1,2) (1,M,2) -> (N,M,2)
    # 排列组合=N*M组(左上内部xy坐标)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2]) # 内部顶点
    # 排列组合=N*M组(右下内部xy坐标)
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # 内部顶点
    # 计算交集区域的宽度和高度
    # .clamp(min=0) 将所有负值截断为0
    # 这一步非常重要，因为当两个边界框没有交集时
    # inter_lowerrights - inter_upperlefts 可能会产生负值
    # 排列组合=N*M组(交集长宽wh)
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    inter_w = inters[:, :, 0]
    inter_h = inters[:, :, 1]
    # 排列组合=N*M组(交集面积)
    inter_areas = inter_w * inter_h
    # ------------------------------------------------- # 

    # 计算并集区域的面积
    # 这里用到广播
    # areas1[N,1] + areas2[1,M] = sum[N,M]
    # inter_areas[N,M]
    # union_areas[N,M]
    union_areas = areas1[:, None] + areas2 - inter_areas
    # 输出(N,M)
    return inter_areas / (union_areas + 1e-6) # 防止除0
