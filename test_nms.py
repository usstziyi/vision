import torch
import matplotlib.pyplot as plt
from common import show_boxes
from common import anchor_shift
from common import filter_boxes_by_nms

torch.set_printoptions(precision=2)     # 保留2位小数
torch.set_printoptions(sci_mode=False)  # 禁用科学计数法

# anchors(NAC, 4)
# 生成锚框(手动):这里相当于已经offset_inverse了
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                        [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], 
                        [0.55, 0.2, 0.9, 0.88]])

# 预测偏移量(手动):这里都设为0，是为了简化问题
pred_offset = torch.tensor([[0, 0, 0, 0], 
                             [0, 0, 0, 0], 
                             [0, 0, 0, 0], 
                             [0, 0, 0, 0]])

# 预测出的边界框
pred_bboxes = anchor_shift(anchors, pred_offset)


# 预测出的类别(手动)            背景  狗  猫
pred_classes = torch.tensor([[0, 3.1, 1.1],
                             [0, 2.2, 0.2],
                             [0, 0.7, 0.3],
                             [0, 0.1, 0.8]])

# 读取图像
img = plt.imread('./img/catdog.jpg')
fig,axes = plt.subplots(1,2)

# 第一幅图:使用NMS之前
axes[0].imshow(img)
axes[0].set_title('Before NMS')
h, w = img.shape[:2]
box_scale = torch.tensor([w, h, w, h])

show_boxes(axes[0], pred_bboxes * box_scale, labels=['dog=0.88', 'dog=0.8', 'dog=0.60', 'cat=0.67'])

# 第二幅图:使用NMS之后
axes[1].imshow(img)
axes[1].set_title('After NMS')

# batch_anchors(1,P*A,4)
batch_anchors = anchors.unsqueeze(dim=0)
# batch_pred_classes(1,P*A*C)
batch_pred_classes = pred_classes.reshape(-1).unsqueeze(dim=0)
# batch_pred_offset(1,P*A*4)
batch_pred_offset = pred_offset.reshape(-1).unsqueeze(dim=0)


batch_boxes_info = filter_boxes_by_nms(batch_anchors, batch_pred_classes, batch_pred_offset, 3)
print(batch_boxes_info)
# 从batch中提取第一个样本的输出
boxes_info = batch_boxes_info[0]
for box_info in boxes_info.detach().numpy():
    if box_info[5] == -1:
        continue
    label = ('dog=', 'cat=')[int(box_info[5])] + f"{box_info[4]:.2f}"
    show_boxes(axes[1], [torch.tensor(box_info[:4]) * box_scale], label)

plt.tight_layout()
plt.show()







