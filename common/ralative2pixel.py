import torch
'''
    将相对坐标转换为像素坐标
'''
# anchors_relative(NAC,4):(xmin,ymin,xmax,ymax)
def relative_to_pixel(boxes_relative, img_pixel_size):
    device = boxes_relative.device
    height = img_pixel_size[-2]
    width = img_pixel_size[-1]
    
    # (w,h,w,h)
    relative = torch.tensor([width, height, width, height], device=device)
    
    # boxes_pixel(NAC,4):(xmin,ymin,xmax,ymax)
    boxes_pixel = boxes_relative * relative
    
    return boxes_pixel
