import torch
from .assign import assign_anchor_to_bbox
from .exchange import box_corner_to_center,box_center_to_corner


'''
    anchors(相对尺寸)到target_anchors(相对尺寸)的偏移量
'''

# 根据锚框和真实框计算偏移量
def offset_boxes(anchors, gt_boxes):
    # anchors: [N, 4] -> (xa, ya, wa, ha)
    # gt_boxes: [N, 4] -> (xb, yb, wb, hb)

    eps = 1e-6
    # 转换为中心点表示
    anchors = box_corner_to_center(anchors)
    gt_boxes = box_corner_to_center(gt_boxes)
    
    # 归一化
    dx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
    dw = torch.log(eps + gt_boxes[:, 2] / anchors[:, 2])
    dh = torch.log(eps + gt_boxes[:, 3] / anchors[:, 3])

    # 标准化
    dx = (dx - 0) * 10
    dy = (dy - 0) * 10
    dw = (dw - 0) * 5
    dh = (dh - 0) * 5

    # [N, 4] -> (dx, dy, dw, dh)
    return torch.stack([dx, dy, dw, dh], dim=1)




'''
    判定每个锚框最可能的类别
    计算每个锚框与这个真实边界框的偏移量
    同时返回掩码
'''
# anchors(1,P*A,4)
# labels(B,G,5):(class,xmin,ymin,xmax,ymax)
def anchor_to_label(anchors, labels):
    """使用真实边界框标记锚框"""
    # B
    batch_size = labels.shape[0]
    # anchors(1,P*A,4)
    anchors = anchors.squeeze(0)
    # P*A   
    num_anchors = anchors.shape[0]
    device = anchors.device
    

    # 存储每个样本中所有锚框是否对应真实边界框的掩码
    list_assigned_mask = [] 
    # 存储每个样本中所有锚框与真实边界框的偏移量
    list_assigned_offset = []
    # 存储每个样本中所有锚框的类别标签
    list_assigned_classes = []
    
    # 一个样本的所有锚框和真实边界框的偏移量
    for i in range(batch_size):
        # labels(B,NGT,5)
        # label(NGT,5)
        label = labels[i, :, :]
        classes = label[:, 0]
        boxes = label[:, 1:]

        # label[:,1:]:(NGT,4):(xmin,ymin,xmax,ymax)
        # anchors(NAC,4):(xmin,ymin,xmax,ymax)
        # anchors_bbox_map(NAC):每个锚框对应的真实边界框索引
        anchors_bbox_map = assign_anchor_to_bbox(boxes, anchors, device)


        # 创建类标签和分配的边界框坐标存储空间
        # assigned_classes(P*A):每个锚框的类别标签
        # assigned_bboxes(P*A,4):每个锚框的真实边界框坐标
        # assigned_classes 被初始化为全 0。这意味着，如果一个锚框没有被任何真实框匹配（即它是负样本），它的标签保持为 0，模型在训练时会将其学习为“背景”。
        assigned_classes = torch.zeros(num_anchors, dtype=torch.long, device=device)          # 用来做分类
        assigned_bboxes = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)   # 用来做回归
        

        # bind大矩阵(anchors_bbox_map的稀疏矩阵)
        # * * * * * 
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *
        # * * * * *




        # 找出正样本的行
        # pos_mask(P*A)
        pos_mask = (anchors_bbox_map >= 0)
        row = torch.nonzero(pos_mask).reshape(-1)
        col = anchors_bbox_map[row]

    
        # 0-bg, 1-c1, 2-c2, ...
        # 而未被分配的锚框（负样本/背景）保持初始值 0
        assigned_classes[row] = classes[col].long() + 1 # 原始标签 0 -> 训练标签 1 （第1类物体）
        assigned_bboxes[row] = boxes[col]
        # 计算锚框到真实框的偏移量
        # assigned_offset(P*A,4)
        assigned_mask = pos_mask.float().unsqueeze(-1).repeat(1, 4)
        assigned_offset = offset_boxes(anchors, assigned_bboxes) * assigned_mask


        # 拼接本batch所有样本的结果
        # list_assigned_classes B(P*A)
        list_assigned_classes.append(assigned_classes)
        # assigned_offset B(P*A*4)
        list_assigned_offset.append(assigned_offset.reshape(-1))
        # assigned_mask B(P*A*4)
        list_assigned_mask.append(assigned_mask.reshape(-1))


    # 压缩本batch所有样本的结果
    batch_assigned_classes = torch.stack(list_assigned_classes, dim=0)
    batch_assigned_offset = torch.stack(list_assigned_offset, dim=0)
    batch_assigned_mask = torch.stack(list_assigned_mask, dim=0)

    # 每个锚框是否对应真实边界框的掩码
    # batch_assigned_classes(B,P*A)
    # batch_assigned_offset(B,P*A*4)
    # batch_assigned_mask(B,P*A*4)

    return (batch_assigned_classes, batch_assigned_offset, batch_assigned_mask)

