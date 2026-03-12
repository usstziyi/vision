import torch
import matplotlib.pyplot as plt
from common import show_boxes,multibox_detection, offset_inverse

torch.set_printoptions(2)  # 恢复默认打印精度
# anchors(NAC, 4)
# 生成锚框(手动):这里相当于已经offset_inverse了
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                        [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], 
                        [0.55, 0.2, 0.9, 0.88]])

# 预测偏移量(手动):这里都设为0，是为了简化问题
offset_preds = torch.tensor([[0, 0, 0, 0], 
                             [0, 0, 0, 0], 
                             [0, 0, 0, 0], 
                             [0, 0, 0, 0]])

# 预测出的边界框
pred_bboxes = offset_inverse(anchors, offset_preds)


# cls_probs(NCLS, NAC)
# 预测出的类别(手动)
pred_classes = torch.tensor([[0,   0,   0,   0],     # 背景的预测概率
                             [0.9, 0.8, 0.7, 0.1],   # 狗的预测概率
                             [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率


# 读取图像
img = plt.imread('./img/catdog.jpg')
fig,axes = plt.subplots(1,2)

# 第一幅图:使用NMS之前
axes[0].imshow(img)
axes[0].set_title('Before NMS')
h, w = img.shape[:2]
box_scale = torch.tensor([w, h, w, h])

show_boxes(axes[0], pred_bboxes * box_scale, labels=['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

# 第二幅图:使用NMS之后
axes[1].imshow(img)
axes[1].set_title('After NMS')

# cls_probs.unsqueeze(dim=0)(1, NCLS, NAC)
# offset_preds.unsqueeze(dim=0)(1, NAC*4)
# anchors.unsqueeze(dim=0)(1, NAC, 4)
# output(1, NAC, 6):(class_id, conf, predicted_bb)
offset_preds = offset_preds.reshape(-1) # (NAC*4)
output = multibox_detection(pred_classes.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
# 从batch中提取第一个样本的输出
output = output[0]

for i in output.detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_boxes(axes[1], [torch.tensor(i[2:]) * box_scale], label)

plt.tight_layout()
plt.show()







