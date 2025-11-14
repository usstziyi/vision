import torch
from d2l import torch as d2l
'''
    anchors(相对尺寸)到target_anchors(相对尺寸)的偏移量
'''
# anchors(NAC,4):(xmin, ymin, xmax, ymax)
# target_anchors(NAC,4):(xmin, ymin, xmax, ymax)
def offset_anchors(anchors, target_anchors, eps=1e-6):
    # 一、将锚框从角点格式转换为中心点格式
    # c_anchors: (NAC, 4):(x, y, w, h)
    # c_target_anchors: (NAC, 4):(x, y, w, h)
    c_anchors = d2l.box_corner_to_center(anchors)
    c_target_anchors = d2l.box_corner_to_center(target_anchors)

    # 二、归一化(中心点和宽高的偏移)
    # offset_xy(NAC, 2):(dx, dy)
    # offset_wh(NAC, 2):(dw, dh)
    offset_xy = (c_target_anchors[:, :2] - c_anchors[:, :2]) / c_anchors[:, 2:]
    offset_wh = torch.log(eps + c_target_anchors[:, 2:] / c_anchors[:, 2:])

    # 三、标准化(中心点和宽高的偏移)
    # offset_xy(NAC, 2):(tx, ty)
    # offset_wh(NAC, 2):(tw, th)
    offset_xy = offset_xy * 10
    offset_wh = offset_wh * 5

    # 四、拼接中心点偏移和宽高偏移，形成最终偏移量张量
    # offset(NAC, 4):(tx, ty, tw, th)
    offset = torch.cat([offset_xy, offset_wh], dim=1)
    return offset



'''
    把偏移量施加到锚框上，得到目标锚框(相对尺寸)
'''
# anchor(NAC,4):(xmin,ymin,xmax,ymax)
# offset(NAC,4):(tx,ty,tw,th)
def offset_inverse(anchors, offset):
    # anchors(NAC,4):(xmin,ymin,xmax,ymax)
    # c_anchors(NAC,4):(x,y,w,h)
    c_anchors = d2l.box_corner_to_center(anchors)
    
    # c_target_anchor_xy(NAC,2):(x,y):施加偏移量后的物理坐标
    # c_target_anchor_wh(NAC,2):(w,h):施加偏移量后的物理尺寸
    # c_target_anchor(NAC,4):(x,y,w,h)
    c_target_anchor_xy = (offset[:, :2] * c_anchors[:, 2:] / 10) + c_anchors[:, :2]
    c_target_anchor_wh = torch.exp(offset[:, 2:] / 5) * c_anchors[:, 2:]
    c_target_anchor = torch.cat((c_target_anchor_xy, c_target_anchor_wh), axis=1)


    # target_anchor(NAC,4):(xmin,ymin,xmax,ymax)
    target_anchor = d2l.box_center_to_corner(c_target_anchor)
    
    return target_anchor
