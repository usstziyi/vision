import torch
from .bind import bind_ground_truth_to_anchor
from .offset import offset_anchors


'''
    使用真实边界框标记锚框
    返回每个锚框的偏移量、掩码、类别标签
'''
# anchors(1,NAC,4)：一个样本中所有锚框坐标
# labels(B,NGT,5):(class, xmin,ymin,xmax,ymax)
# labels 传入的是批次样本的真实边界框标签和坐标
# 而 anchors 传入的是单个样本的锚框集合。
def anchor_label(anchors, labels):
    """使用真实边界框标记锚框"""
    # B
    batch_size = labels.shape[0]
    # anchors(NAC,4)：一个样本中所有锚框坐标
    anchors = anchors.squeeze(0)
    # NAC
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
        classes = label[:, 0].long()
        boxes = label[:, 1:]

        # label[:,1:]:(NGT,4):(xmin,ymin,xmax,ymax)
        # anchors(NAC,4):(xmin,ymin,xmax,ymax)
        # anchors_bbox_map(NAC):每个锚框对应的真实边界框索引
        anchors_bbox_map = assign_anchor_to_bbox(boxes, anchors, device)


        # 创建类标签和分配的边界框坐标存储空间
        # class_labels(NAC):每个锚框的类别标签
        # assigned_bb(NAC,4):每个锚框的真实边界框坐标
              # 用来做掩码
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
        assigned_mask = (anchors_bbox_map > 0).long()
        row = torch.nonzero(assigned_mask).reshape(-1)
        col = anchors_bbox_map[row]
    
        # 0-bg, 1-c1, 2-c2, ...
        assigned_classes[row] = classes[col] + 1
        assigned_bboxes[row] = boxes[col]
        assigned_offset = offset_boxes(anchors, assigned_bboxes) * assigned_mask.unsqueeze(-1)


  
    
        # 拼接本batch所有样本的结果
        # bbox_mask.reshape(-1):(4*NAC,):(1,1,1,1,0,0,0,0,1,1,1,1,...)
        list_assigned_mask.append(assigned_mask.reshape(-1))
        # offset.reshape(-1):(4*NAC,):(tx1,ty1,tw1, th1, tx2, ty2, tw2, th2, ...)
        list_assigned_offset.append(assigned_offset)
        # class_labels(NAC,):(c1,c2,c3,0,0,0,c4,c5,...)
        list_assigned_classes.append(assigned_classes)

    # 压缩本batch所有样本的结果
    # batch_assigned_offset(B,4*NAC):(tx1,ty1,tw1, th1, tx2, ty2, tw2, th2, ...)
    batch_assigned_offset = torch.stack(list_assigned_offset, dim=0)
    # batch_assigned_mask(B,4*NAC):(1,1,1,1,0,0,0,0,1,1,1,1,...)
    batch_assigned_mask = torch.stack(list_assigned_mask, dim=0)
    # batch_assigned_classes(B,NAC):(c1,c2,c3,0,0,0,c4,c5,...)
    batch_assigned_classes = torch.stack(list_assigned_classes, dim=0)
    # 每个锚框是否对应真实边界框的掩码
    # 偏移量：batch_assigned_offset(B,4*NAC):(tx1,ty1,tw1, th1, tx2, ty2, tw2, th2, ...)
    # 掩码：batch_assigned_mask(B,4*NAC):(1,1,1,1,0,0,0,0,1,1,1,1,...)
    # 类别：batch_assigned_classes(B,NAC):(c1,c2,c3,0,0,0,c4,c5,...)
    return (batch_assigned_offset, batch_assigned_mask, batch_assigned_classes)
