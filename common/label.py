import torch
from .bind import bind_ground_truth_to_anchor
from .offset import offset_anchors


'''
    使用真实边界框标记锚框
    返回每个锚框的偏移量、掩码、类别标签
'''


# anchors(NAC,4)：一个样本中所有锚框坐标
# labels (NGT,5):(class,xmin,ymin,xmax,ymax)
def anchors_label(anchors, labels):
    device = anchors.device

    # NAC
    num_anchors = anchors.shape[0]   

    # labels(NGT,5):(class,xmin,ymin,xmax,ymax)
    # label(NGT,)
    label = labels[:, 0]
    # gts:(NGT,4):(xmin,ymin,xmax,ymax)
    gts = labels[:, 1:]

    # gts(NGT,4):(xmin,ymin,xmax,ymax)
    # anchors(NAC,4):(xmin,ymin,xmax,ymax)
    # anchors_gt_map(NAC,):-1表示anchor未bind到gt,anchor是负样本
    anchors_gt_map = bind_ground_truth_to_anchor(gts, anchors, device)

    # (anchors_gt_map >= 0).float():float(NAC,)
    # .unsqueeze(-1):(NAC,1)
    # .repeat(1, 4):float(NAC,4)
    # positive_mask(NAC,4):每个锚框是否对应真实边界框的掩码
    # [[1,1,1,1],
    #  [0,0,0,0],
    #  [1,1,1,1]]
    positive_mask = ((anchors_gt_map >= 0).float().unsqueeze(-1)).repeat(1, 4)

    # 创建类标签和分配的边界框坐标存储空间
    # anchors_label(NAC):每个锚框的类别标签
    # anchors_gt(NAC,4):每个锚框的真实边界框坐标
    anchors_label = torch.zeros(num_anchors, dtype=torch.long, device=device)
    anchors_gt = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
    

    # positive_ac_id(P,1)->(P,)
    positive_ac_id = torch.nonzero(anchors_gt_map >= 0).reshape(-1)
    # positive_gt_id(P,):每个锚框对应的真实边界框索引
    positive_gt_id = anchors_gt_map[positive_ac_id]
    # 填充类别Class
    # 0-bg, 1-c1, 2-c2, ...
    # 因为label中的类别不包括背景类
    # 所以label中的类别要+1，和本第的规则匹配
    anchors_label[positive_ac_id] = label[positive_gt_id].long() + 1
    # 填充真实边界框坐标(xmin,ymin,xmax,ymax)
    anchors_gt[positive_ac_id] = gts[positive_gt_id, :]   
    
    # 一个样本中所有锚框和真实边界框的偏移量
    # anchors(NAC,4):(xmin,ymin,xmax,ymax)
    # anchors_gt(NAC,4):(xmin,ymin,xmax,ymax)
    # gt_mask(NAC,4):每个锚框是否对应真实边界框的掩码
    # 一个ac对一个gt，计算ac到gt的偏移量
    # 其中有的ac(负样本)对的空gt为(0,0,0,0)，所以最后要乘以positive_mask
    # 最后得到的offset都是正样本的偏移量
    # offset(NAC,4):(dx,dy,dw,dh)
    offset = offset_anchors(anchors, anchors_gt) * positive_mask

    # offset(4*NAC,):(tx1,ty1,tw1, th1, tx2, ty2, tw2, th2, ...)
    # positive_mask(4*NAC,):(1,1,1,1,0,0,0,0,1,1,1,1,...)
    # anchors_label(NAC,):(c1,c2,c3,c4,c5,...)
    offset = offset.reshape(-1)
    positive_mask = positive_mask.reshape(-1)
    anchors_label = anchors_label.reshape(-1)


    return (offset, positive_mask, anchors_label)  
