import torch
from .assign import assign_anchor_to_bbox


'''
    anchors(相对尺寸)到target_anchors(相对尺寸)的偏移量
'''

# 根据锚框和真实框计算偏移量
def offset_boxes(anchors, gt_boxes):
    # anchors: [N, 4] -> (xa, ya, wa, ha)
    # gt_boxes: [N, 4] -> (xb, yb, wb, hb)

    eps = 1e-6

    # 归一化
    dx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2]
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3]
    dw = torch.log(eps + gt_boxes[:, 2] / anchors[:, 2])
    dh = torch.log(eps + gt_boxes[:, 3] / anchors[:, 3])

    # 标准化
    dx = (dx - 0) / 0.1
    dy = (dy - 0) / 0.1
    dw = (dw - 0) / 0.2
    dh = (dh - 0) / 0.2

    # [N, 4] -> (dx, dy, dw, dh)
    return torch.stack([dx, dy, dw, dh], dim=1)


'''
    判定每个锚框最可能的类别
    计算每个锚框与这个真实边界框的偏移量
    同时返回掩码
'''
# anchors(1,NAC,4)：一个样本中所有锚框坐标
# labels(B,NGT,5):(class, xmin,ymin,xmax,ymax)
# labels 传入的是批次样本的真实边界框标签和坐标
# 而 anchors 传入的是单个样本的锚框集合。
def anchor_to_label(anchors, labels):
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
        assigned_mask = (anchors_bbox_map >= 0).long()
        row = torch.nonzero(assigned_mask).reshape(-1)
        col = anchors_bbox_map[row]

    
        # 0-bg, 1-c1, 2-c2, ...
        # 而未被分配的锚框（负样本/背景）保持初始值 0
        assigned_classes[row] = classes[col] + 1
        assigned_bboxes[row] = boxes[col]
        # 计算锚框到真实框的偏移量
        assigned_offset = offset_boxes(anchors, assigned_bboxes) * assigned_mask.unsqueeze(-1)


  
    
        # 拼接本batch所有样本的结果
        # class_labels(NAC,):(c1,c2,c3,0,0,0,c4,c5,...)
        list_assigned_classes.append(assigned_classes)
        # offset.reshape(-1):(4*NAC,):(tx1,ty1,tw1, th1, tx2, ty2, tw2, th2, ...)
        list_assigned_offset.append(assigned_offset)
        # bbox_mask.reshape(-1):(4*NAC,):(1,1,1,1,0,0,0,0,1,1,1,1,...)
        list_assigned_mask.append(assigned_mask.reshape(-1))


    # 压缩本batch所有样本的结果
    # batch_assigned_classes(B,NAC):(c1,c2,c3,0,0,0,c4,c5,...)
    batch_assigned_classes = torch.stack(list_assigned_classes, dim=0)
    # batch_assigned_offset(B,4*NAC):(tx1,ty1,tw1, th1, tx2, ty2, tw2, th2, ...)
    batch_assigned_offset = torch.stack(list_assigned_offset, dim=0)
    # batch_assigned_mask(B,4*NAC):(1,1,1,1,0,0,0,0,1,1,1,1,...)
    batch_assigned_mask = torch.stack(list_assigned_mask, dim=0)

    # 每个锚框是否对应真实边界框的掩码
    # 类别：batch_assigned_classes(B,NAC):(c1,c2,c3,0,0,0,c4,c5,...)
    # 偏移量：batch_assigned_offset(B,4*NAC):(tx1,ty1,tw1, th1, tx2, ty2, tw2, th2, ...)
    # 掩码：batch_assigned_mask(B,4*NAC):(1,1,1,1,0,0,0,0,1,1,1,1,...)
    print("batch_assigned_classes:",batch_assigned_classes.shape)
    print("batch_assigned_offset:",batch_assigned_offset.shape)
    print("batch_assigned_mask:",batch_assigned_mask.shape)
    return (batch_assigned_classes, batch_assigned_offset, batch_assigned_mask)



# 把偏移量施加到锚框上，得到预测框
# 注：这个函数是预测阶段用的，不是训练阶段用的
# offsets是预测的偏移量，因为预测阶段没有标签可用
# 这个函数返回锚框+预测后的结果
def offset_inverse(anchors, offsets):
    """
    将偏移量还原为边界框坐标。
    
    Args:
        anchors: Tensor [N, 4], 格式 (xa, ya, wa, ha)
        offsets: Tensor [N, 4], 格式 (dx, dy, dw, dh) (已标准化)
        
    Returns:
        gt_boxes: Tensor [N, 4], 格式 (xb, yb, wb, hb)
    """
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

    # 堆叠返回 [N, 4] -> (xb, yb, wb, hb)
    gt_boxes = torch.stack([xb, yb, wb, hb], dim=1)
    
    return gt_boxes
