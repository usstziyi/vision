import torch

'''
    根据输入图像尺寸、缩放比例列表、宽高比列表生成多个锚框
'''

# data:输入1张图像
# scales:缩放列表
# ratios:宽高比列表
def generate_anchors(data, scales, ratios):
    device = data.device

    num_scales = len(scales)
    num_ratios = len(ratios)

    # 列表转换为张量
    # scales_tensor(num_scales,):(s1,s2,...,sn)
    # ratios_tensor(num_ratios,):(r1,r2,...,rm)
    scales_tensor = torch.tensor(scales, device=device)
    ratios_tensor = torch.tensor(ratios, device=device)
    # 每个像素中心生成的锚框数量
    num_anchors_per_pixel = (num_scales + num_ratios - 1)

    # 一、像素坐标系：物理坐标系
    # .----->x
    # |
    # y
    # data_height:y轴方向上的像素数
    # data_width:x轴方向上的像素数
    data_height = data.shape[-2]
    data_width = data.shape[-1]

    # 中心点偏移半个像素
    offset_h, offset_w = 0.5, 0.5 # 单位：像素

    # x、y轴方向上的所有像素中心坐标
    center_x_tensor = torch.arange(data_width, device=device) + offset_w
    center_y_tensor = torch.arange(data_height, device=device) + offset_h

    # 根据缩放长宽比生成的锚框的像素长宽
    # anchors_width(num_anchors_per_pixel,)
    # anchors_height(num_anchors_per_pixel,)
    anchors_width = torch.cat((data_height * scales_tensor * torch.sqrt(ratios_tensor[0]), data_height * scales_tensor[0] * torch.sqrt(ratios_tensor[1:])))
    anchors_height = torch.cat((data_height * scales_tensor / torch.sqrt(ratios_tensor[0]), data_height * scales_tensor[0] / torch.sqrt(ratios_tensor[1:])))



    # 二、锚框坐标系：逻辑坐标系（归一化坐标）(定位坐标)(不管原图片是否是正方形)
    # .----->1
    # |
    # 1
    # data_relative_height:y轴方向上的逻辑长度
    # data_relative_width:x轴方向上的逻辑长度
    data_relative_height = 1.0
    data_relative_width = 1.0

    # x、y轴方向上的所有像素中心坐标的归一化逻辑坐标
    center_relative_x_tensor = center_x_tensor / data_width
    center_relative_y_tensor = center_y_tensor / data_height

    # x,y网格化
    # mesh_y(h,w)
    # mesh_x(h,w)
    mesh_y, mesh_x = torch.meshgrid(center_relative_y_tensor, center_relative_x_tensor, indexing='ij')          
    # 展平后：h * w 个
    # mesh_y = [0.5, 0.5, 0.5, 1.5, 1.5, 1.5, 2.5, 2.5, 2.5, 3.5, 3.5, 3.5]  # 形状: [12]
    # mesh_x = [0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 0.5, 1.5, 2.5]  # 形状: [12]
    # y行先行
    # 避免了两层for循环，直接利用广播机制生成了所有锚框的中心坐标
    mesh_y, mesh_x = mesh_y.reshape(-1), mesh_x.reshape(-1)

    # 根据函数参数，每个中心点都生成“boxes_per_pixel”个锚框
    # num_anchors_per_pixel = num_scales + num_ratios - 1
    # []负责传递参数，不增加维度
    # [mesh_x] [mesh_y] [mesh_x] [mesh_y]
    # [mesh_x] [mesh_y] [mesh_x] [mesh_y]
    # [mesh_x] [mesh_y] [mesh_x] [mesh_y]
    # [mesh_x] [mesh_y] [mesh_x] [mesh_y]
    # [mesh_x] [mesh_y] [mesh_x] [mesh_y]
    # center_relative_xyxy(h*w, 4)
    # [mesh_x, mesh_y, mesh_x, mesh_y]
    # [mesh_x, mesh_y, mesh_x, mesh_y]
    # [mesh_x, mesh_y, mesh_x, mesh_y]
    # [mesh_x, mesh_y, mesh_x, mesh_y]
    # [mesh_x, mesh_y, mesh_x, mesh_y]
    center_relative_xyxy = torch.stack([mesh_x, mesh_y, mesh_x, mesh_y], dim=1)
    # center_relative_xyxy(num_anchors_per_pixel*h*w, 4)
    center_relative_xyxy = center_relative_xyxy.repeat_interleave(num_anchors_per_pixel, dim=0)



    # 锚框的相对长宽
    # anchors_relative_width(num_anchors_per_pixel,)
    # anchors_relative_height(num_anchors_per_pixel,)
    anchors_relative_width = anchors_width / data_width
    anchors_relative_height = anchors_height / data_height


    # anchors_relative_whwh(num_anchors_per_pixel, 4)
    anchors_relative_whwh = torch.stack([-anchors_relative_width, -anchors_relative_height, anchors_relative_width, anchors_relative_height], dim=1)
    # anchors_relative_whwh(num_anchors_per_pixel*h*w, 4)
    anchors_relative_whwh = anchors_relative_whwh.repeat(data_height * data_width, 1) / 2


    # 五、中心坐标+长宽偏移量=左上角坐标、右下角坐标
    # center_relative_xyxy          (num_anchors_per_pixel*h*w, 4)
    # anchors_relative_whwh         (num_anchors_per_pixel*h*w, 4)
    # anchors_relative              (num_anchors_per_pixel*h*w, 4)
    # (xmin, ymin, xmax, ymax)
    anchors_relative = center_relative_xyxy + anchors_relative_whwh
    # 输出：(1, bpp*h*w, 4)
    return anchors_relative.unsqueeze(0)



    # (bpp,)
    # w = [1.0, 0.8, 1.2, 0.6, 1.5]  # 5个锚框的宽度
    # h = [1.0, 1.2, 0.8, 1.5, 0.6]  # 5个锚框的高度
    # 执行 torch.stack((-w, -h, w, h)) 后，会得到一个形状为(4, 5)的张量：
    # (4,bpp)
    # tensor([[-1.0, -0.8, -1.2, -0.6, -1.5],   # -w
    #         [-1.0, -1.2, -0.8, -1.5, -0.6],   # -h
    #         [ 1.0,  0.8,  1.2,  0.6,  1.5],   # w
    #         [ 1.0,  1.2,  0.8,  1.5,  0.6]])  # h
    # 转置
    # (bpp,4)   -w    -h    w     h
    # tensor([[-1.0, -1.0,  1.0,  1.0],   # 第1个锚框的偏移量
    #         [-0.8, -1.2,  0.8,  1.2],   # 第2个锚框的偏移量
    #         [-1.2, -0.8,  1.2,  0.8],   # 第3个锚框的偏移量
    #         [-0.6, -1.5,  0.6,  1.5],   # 第4个锚框的偏移量
    #         [-1.5, -0.6,  1.5,  0.6]])  # 第5个锚框的偏移量
    # .repeat(in_height * in_width, 1)
    # (bpp*h*w,4)
    # 这一步的目的是为图像中的每个像素点复制相同的锚框尺寸信息。例如，如果图像大小为2×2（4个像素点），则结果为形状(20, 4)的张量
    # tensor([[-1.0, -1.0,  1.0,  1.0],   # 第1个锚框，第1个像素点
    #         [-0.8, -1.2,  0.8,  1.2],   # 第2个锚框，第1个像素点
    #         [-1.2, -0.8,  1.2,  0.8],   # 第3个锚框，第1个像素点
    #         [-0.6, -1.5,  0.6,  1.5],   # 第4个锚框，第1个像素点
    #         [-1.5, -0.6,  1.5,  0.6],   # 第5个锚框，第1个像素点

    #         [-1.0, -1.0,  1.0,  1.0],   # 第1个锚框，第2个像素点
    #         [-0.8, -1.2,  0.8,  1.2],   # 第2个锚框，第2个像素点
    #         ...                         # 依此类推
    #         [-1.5, -0.6,  1.5,  0.6]])  # 第5个锚框，第4个像素点
    # 总个数：(num_sizes + num_ratios - 1) * in_height * in_width = (n*m-1)*h*w