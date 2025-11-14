from common import generate_anchors
import torch
from d2l import torch as d2l

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
    print(anchors[250,250,0,:])


if __name__ == '__main__':
    main()