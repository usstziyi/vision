from common import generate_anchors, show_anchors, relative_to_pixel
import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

torch.set_printoptions(2) 


def main():
    img = d2l.plt.imread('./img/catdog.jpg')
    h, w = img.shape[:2]
    print(h, w)
    data = torch.rand(size=(1, 3, h, w))


    scales = [0.75, 0.5, 0.25]
    ratios = [1, 2, 0.5]
    Y = generate_anchors(data, scales, ratios)
    print(Y.shape)

    anchors = Y.reshape(h,w,5,4)
    print(anchors[250,250,:,:])


    # 将锚框从相对坐标转换为像素坐标
    print(data.shape)
    anchors_pixel = relative_to_pixel(anchors, data.shape)
    print(anchors_pixel[250, 250, :, :])

    # d2l.set_figsize()
    fig = plt.imshow(img)
    # 为每个锚框添加标签，方便观察不同尺度和比例的锚框效果
    labels = ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2', 's=0.75, r=0.5']
    show_anchors(fig.axes, anchors_pixel[250, 250, :, :], labels)
    plt.title('(250,250)')
    plt.show()


if __name__ == '__main__':
    main()