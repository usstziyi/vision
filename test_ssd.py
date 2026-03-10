import torch
import torchvision
from torch import nn
from torch.nn import functional as F

# 类别预测层
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs,                       # 输入通道数
                     num_anchors * (num_classes + 1),  # 输出通道数：每个锚框预测 num_classes+1 个值
                     kernel_size=3,                    # 卷积核大小
                     padding=1)                        # 填充大小，保持特征图尺寸不变


# 边界框预测层
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs,                       # 输入通道数
                     num_anchors * 4,                  # 输出通道数：每个锚框预测 4 个值（偏移量）
                     kernel_size=3,                    # 卷积核大小
                     padding=1)                        # 填充大小，保持特征图尺寸不变


cls_model1 = cls_predictor(8, 5, 10)
cls_model2 = cls_predictor(16, 3, 10)
X1 = torch.randn(2, 8, 20, 20)
X2 = torch.randn(2, 16, 10, 10)
Y1 = cls_model1(X1)
Y2 = cls_model2(X2)
print(Y1.shape)
print(Y2.shape)


# 转置
def transpose_output(preds):
    return preds.permute(0, 2, 3, 1)

# 展平
def flatten_pred(preds):
    return torch.flatten(transpose_output(preds), start_dim=1) # 从最高维开始抽丝

# 拼接
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], axis=1)

print(concat_preds([Y1,Y2]).shape)


