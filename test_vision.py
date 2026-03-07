import torch
import matplotlib.pyplot as plt
from common import generate_anchors, show_boxes, relative_to_pixel

# 设置打印选项，保留两位小数
torch.set_printoptions(2) 

"""
    这个demo展示了如何使用generate_anchors函数生成锚框，并将其可视化。
    没有涉及到iou计算。
    没有涉及到nms。
    没有涉及到锚框的筛选。
    没有涉及到目标检测。
"""


def main():
    # 读取图像
    img = plt.imread('./img/catdog.jpg')
    h, w = img.shape[:2]
    print(h, w)
    data = torch.rand(size=(1, 3, h, w))


    scales = [0.75, 0.5, 0.25]
    ratios = [1, 2, 0.5]
    anchors_relative = generate_anchors(data, scales, ratios)
    anchors_relative = anchors_relative.reshape(h,w,-1,4)
    print(anchors_relative.shape)

    # 归一化坐标转换为像素坐标
    anchors_pixel = relative_to_pixel(anchors_relative, data.shape)
    anchors_pixel = anchors_pixel.reshape(h,w,-1,4)
    anchors_one_point = anchors_pixel[250, 250, :, :]
    print(anchors_one_point)


    # 创建画布
    fig,axes = plt.subplots(1,1)
    # 显示背景图
    axes.imshow(img)

    # 为每个锚框添加标签，方便观察不同尺度和比例的锚框效果
    labels = ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5']
    show_boxes(axes, anchors_one_point, labels)
    plt.show()


if __name__ == '__main__':
    main()