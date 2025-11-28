from common import generate_fmap_anchors
from common import relative_to_pixel
from common import show_anchors
from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt

torch.set_printoptions(2) 

def main():
    # 读取图像
    img = d2l.plt.imread('./img/catdog.jpg')
    h, w = img.shape[:2]
    print(h, w)

    # 拟定特征图的宽度和高度
    fmap_w, fmap_h = 4, 4
    scales = [0.15]
    ratios = [1, 2, 0.5]
    num_anchors = len(scales) + len(ratios) - 1
    # 以特征图的每个单元为中心生成锚框
    # anchors(NAC,4):(xmin,ymin,xmax,ymax)
    anchors = generate_fmap_anchors(fmap_w, fmap_h, scales, ratios)
    # anchors(h,w,num_anchors,4)
    anchors = anchors.reshape(fmap_h, fmap_w, num_anchors, 4)

    # 将锚框从相对坐标转换为像素坐标，坐标放大映射
    # anchors_pixel(h,w,num_anchors,4)
    anchors_pixel = relative_to_pixel(anchors, img.shape[:2])

    # 可视化锚框
    fig = plt.imshow(img)
    show_anchors(fig.axes, anchors_pixel[0,0,:,:])
    show_anchors(fig.axes, anchors_pixel[1,1,:,:])
    show_anchors(fig.axes, anchors_pixel[2,2,:,:])
    show_anchors(fig.axes, anchors_pixel[3,3,:,:])

    plt.show()


if __name__ == '__main__':
    main()