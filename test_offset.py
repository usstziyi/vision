import torch
import matplotlib.pyplot as plt
from common import show_boxes, anchor_label

# ground_truth(B,5):(class_index,xmin,ymin,xmax,ymax)
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])
# anchors(NAC,4):(xmin,ymin,xmax,ymax)
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], 
                        [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98],
                        [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])
# 读取图像
img = plt.imread('./img/catdog.jpg')
h, w = img.shape[:2]



fig = plt.imshow(img)
bbox_scale = torch.tensor((w, h, w, h))
# 乘以 bbox_scale 将归一化坐标转换为实际的像素坐标
show_boxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_boxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])

# 提升维度，将锚框和真实边界框转换为批量格式
anchors = anchors.unsqueeze(dim=0)
ground_truth = ground_truth.unsqueeze(dim=0)

labels = anchor_label(anchors, ground_truth)

plt.show()

print(labels[0])
print(labels[1])
print(labels[2])
