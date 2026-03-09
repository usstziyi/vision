import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
from common import show_boxes, anchor_label


anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], 
                        [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], 
                        [0.55, 0.2, 0.9, 0.88]])



offset_preds = torch.tensor([0] * anchors.numel())


# cls_probs(C,N):每个类别上每个锚框的预测概率
cls_probs = torch.tensor([[0,   0,   0,   0],              # 背景的预测概率
                          [0.9, 0.8, 0.7, 0.1],   # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率


# 读取图像
img = plt.imread('./img/catdog.jpg')
fig = plt.imshow(img)
h, w = img.shape[:2]
box_scale = torch.tensor([w, h, w, h])

# 还没有做NMS，所以所有锚框都被保留了
show_boxes(fig.axes, anchors * box_scale,['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])



plt.show()



