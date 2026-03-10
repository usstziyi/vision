import torch
import matplotlib.pyplot as plt
from common import generate_anchors_by_feature_map, show_boxes


img = plt.imread('./img/catdog.jpg')
h, w = img.shape[:2]
fig, axes = plt.subplots(2, 2)

# 第一个特征图(1x1)
axes[0, 0].imshow(img)
anchors = generate_anchors_by_feature_map(fmap_w=1, fmap_h=1, scales=[0.15])
box_scale = torch.tensor((w, h, w, h))
show_boxes(axes[0, 0], anchors[0] * box_scale)

# 第二个特征图(2x2)
axes[0, 1].imshow(img)
anchors = generate_anchors_by_feature_map(fmap_w=2, fmap_h=2, scales=[0.15])
box_scale = torch.tensor((w, h, w, h))
show_boxes(axes[0, 1], anchors[0] * box_scale)

# 第三个特征图(3x3)
axes[1, 0].imshow(img)
anchors = generate_anchors_by_feature_map(fmap_w=3, fmap_h=3, scales=[0.15])
box_scale = torch.tensor((w, h, w, h))
show_boxes(axes[1, 0], anchors[0] * box_scale)

# 第四个特征图(4x4)
axes[1, 1].imshow(img)
anchors = generate_anchors_by_feature_map(fmap_w=4, fmap_h=4, scales=[0.15])
box_scale = torch.tensor((w, h, w, h))
show_boxes(axes[1, 1], anchors[0] * box_scale)


plt.show()