import torch
'''
    将相对坐标转换为像素坐标
'''
# anchors_relative(NAC,4):(xmin,ymin,xmax,ymax)
def relative_to_pixel(anchors_relative, data_pixel_size):
    device = anchors_relative.device
    height = data_pixel_size[-2]
    width = data_pixel_size[-1]
    
    # (w,h,w,h)
    relative = torch.tensor([width, height, width, height], device=device)
    
    # anchors_pixel(NAC,4):(xmin,ymin,xmax,ymax)
    anchorss_pixel = anchors_relative * relative
    
    return anchorss_pixel
