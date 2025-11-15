import torch

"""
    计算两组锚框的交并比
"""
# anchors1(NAC1,4):(xmin, ymin, xmax, ymax)
# anchors2(NAC2,4):(xmin, ymin, xmax, ymax)
def iou(anchors1, anchors2):
    # 这种计算方式基于边界框的标准表示方法：[xmin, ymin, xmax, ymax]，其中：
    # - anchors1[:, 0] 是左上角x坐标
    # - anchors1[:, 1] 是左上角y坐标
    # - anchors1[:, 2] 是右下角x坐标
    # - anchors1[:, 3] 是右下角y坐标
    # anchors_area(NAC)=(xmax-xmin)*(ymax-ymin)
    anchors_area = lambda anchors: ((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1]))

    # areas1：(NAC1,)
    # areas2：(NAC2,)
    areas1 = anchors_area(anchors1)
    areas2 = anchors_area(anchors2)
    # anchors1[:, None, :2] 的形状为 (NAC1, 1,    2) :(xmin, ymin)
    # anchors2[:, :2]    广播后形状为 (1，   NAC2, 2) :(xmin, ymin)
    # 广播(NAC1,NAC2,2):(x左上,y左上)
    inter_upperlefts = torch.max(anchors1[:, None, :2], anchors2[:, :2]) # 内部左上角顶点
    # anchors1[:, None, 2:] 的形状为 (NAC1, 1, 2) :(xmax, ymax)        
    # anchors2[:, 2:]       的形状为 (NAC2, 2)    :(xmax, ymax)
    # 广播(NAC1,NAC2,2):(x右下,y右下)
    inter_lowerrights = torch.min(anchors1[:, None, 2:], anchors2[:, 2:]) # 内部右下角顶点
    # 计算交集区域的宽度和高度
    # .clamp(min=0) 将所有负值截断为0
    # 这一步非常重要，因为当两个边界框没有交集时
    # inter_lowerrights - inter_upperlefts 可能会产生负值
    # inters(NAC1,NAC2,2):(x右下-x左上,y右下-y左上)
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # 计算交集区域的面积
    # inters[:, :, 0]:宽=(x右下-x左上)
    # inters[:, :, 1]:高=(y右下-y左上)
    # inter_areas(NAC1,NAC2):交集面积=宽*高
    inter_areas = inters[:, :, 0] * inters[:, :, 1]

    # 计算并集区域的面积
    # (NAC1,1)+(NAC2)=(NAC1,1)+(1,NAC2)=(NAC1,NAC2)
    # (NAC1,NAC2)-(NAC1,NAC2)
    # union_areas(NAC1,NAC2)
    union_areas = areas1[:, None] + areas2 - inter_areas
    # 输出(NAC1,NAC2)  
    return inter_areas / union_areas