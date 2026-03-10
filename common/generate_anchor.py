import torch
# sizes:缩放比例列表
# ratios:宽高比列表
# 生成以每个像素为中心具有不同形状的锚框
def generate_anchors(picture, sizes, ratios):
    # picture(B,C,H,W)
    device, num_sizes, num_ratios = picture.device, len(sizes), len(ratios)
    # size_tensor：不同缩放比例的个数
    size_tensor = torch.tensor(sizes, device=device)
    # ratio_tensor：不同宽高比的个数
    ratio_tensor = torch.tensor(ratios, device=device)


    # 一、像素坐标系：物理坐标系
    # .----->x
    # |
    # y
    # picture_h:y轴方向上的像素数
    # picture_w:x轴方向上的像素数
    picture_h, picture_w = picture.shape[-2:]   
    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5 # 单位：像素

    # 二、锚框坐标系：逻辑坐标系（归一化坐标）(定位坐标)
    # .----->1
    # |
    # 1
    # 给像素坐标系覆上锚框坐标系


    # 定锚框中心物理坐标
    center_y = (torch.arange(picture_h, device=device) + offset_h)
    center_x = (torch.arange(picture_w, device=device) + offset_w)
    # 归一化锚框中心物理坐标到逻辑坐标
    center_y_logical = center_y / picture_h
    center_x_logical = center_x / picture_w

    # 组合所有中心点的逻辑坐标
    shift_y, shift_x = torch.meshgrid(center_y_logical, center_x_logical, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 第一步：[[x,y,x,y]]:(h*w,4)
    # 第二步：独立复制第0维度:(bpp*h*w,4)
    # bbox_center_logic_grid(bpp*h*w, 4)是用来生成锚框的中心逻辑坐标(原材料)
    # 这里为什么使用了两组中心坐标：因为后面需要算左上和右下的坐标，需要两组中心坐标来计算
    bbox_center_logic_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave((num_sizes + num_ratios - 1), dim=0)

    # ---------------------------------------------------------------------------- # 

    # 根据size和ratio计算锚框的实际物理长度
    # w(bpp,)=(n+m-1,)
    # (s0,r0),(s1,r0),(s3,r0),...,(sn-1,r0):n
    # (s0,r1),(s0,r2),(s0,r3),...,(s0,rm-1):m-1
    bbox_w = torch.cat((
        size_tensor * torch.sqrt(ratio_tensor[0]),
        sizes[0] * torch.sqrt(ratio_tensor[1:])
    )) * picture_h
    # h(bpp,)=(n+m-1,)
    bbox_h = torch.cat((
        size_tensor / torch.sqrt(ratio_tensor[0]),
        sizes[0] / torch.sqrt(ratio_tensor[1:])
    )) * picture_h

    # 归一化，得到锚框的归一化长度
    bbox_w_logic = bbox_w / picture_w
    bbox_h_logic = bbox_h / picture_h

    # 取半，得到锚框的归一化半长
    bbox_w_half_logic = bbox_w_logic / 2 # len(bbox_w_half_logic)=n+m-1
    bbox_h_half_logic = bbox_h_logic / 2 # len(bbox_h_half_logic)=n+m-1


    # bbox_w_h_half_logic(bpp*h*w, 4)
    # 第一步：[[-w,-h,w,h]]:(n+m-1,4)
    # 第二步：整体复制in_height * in_width次，得到(bpp*h*w,4)
    bbox_w_h_half_logic = torch.stack([-bbox_w_half_logic, -bbox_h_half_logic, bbox_w_half_logic, bbox_h_half_logic], dim=1).repeat(picture_h * picture_w, 1)


    # 中心坐标+长宽偏移量=左上角坐标、右下角坐标
    # bbox_center_logic_grid            (bpp*h*w, 4)
    # bbox_w_h_half_logic               (bpp*h*w, 4)
    # bbox_logic                        (bpp*h*w, 4)
    # (左上角x, 左上角y, 右下角x, 右下角y)
    bbox_logic = bbox_center_logic_grid + bbox_w_h_half_logic
    # 输出：(1, bpp*h*w, 4)
    return bbox_logic.unsqueeze(0)


# 通过特征图生成锚框(内部还是调用generate_anchors)
def generate_anchors_by_feature_map(fmap_w, fmap_h, scales, ratios=[1, 2, 0.5]):
    # 前两个维度上的值不影响输出
    # 等效特征图(B, C, h, w)
    fmap = torch.zeros((1, 3, fmap_h, fmap_w)) # 影子张量
    # 在特征图上生成锚框:归一化坐标
    anchors = generate_anchors(fmap, scales, ratios)
    # 输出：(1, bpp*h*w, 4)
    return anchors