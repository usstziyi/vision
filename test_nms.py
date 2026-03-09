import torch
import matplotlib.pyplot as plt
from common import show_boxes, anchor_label, multibox_detection

torch.set_printoptions(2)  # 恢复默认打印精度


anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], 
                        [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], 
                        [0.55, 0.2, 0.9, 0.88]])



offset_preds = torch.tensor([0] * anchors.numel())


# cls_probs(C,N):每个类别上每个锚框的预测概率
# 此处的0时概率，不是类别
cls_probs = torch.tensor([[0,   0,   0,   0],     # 背景(-1)的预测概率
                          [0.9, 0.8, 0.7, 0.1],   # 狗(0)的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫(1)的预测概率


# 读取图像
img = plt.imread('./img/catdog.jpg')
fig,axes = plt.subplots(2,1)

# 第一幅图
axes[0].imshow(img)
axes[0].set_title('Before NMS')
h, w = img.shape[:2]
box_scale = torch.tensor([w, h, w, h])

# 还没有做NMS，所以所有锚框都被保留了
show_boxes(axes[0], anchors * box_scale,['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

# 做NMS
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)

# 从batch中提取第一个样本的检测结果
output = output[0]
print(output)


# 第二幅图
axes[1].imshow(img)
axes[1].set_title('After NMS')
for i in output.detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_boxes(axes[1], [torch.tensor(i[2:]) * box_scale], label)

plt.tight_layout()
plt.show()






