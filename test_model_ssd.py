import torch
from torch import nn
from common import generate_anchors

# 拼接多尺寸预测结果
# return(B,(H*W+...)*C)
def concat_preds(preds):
    list_preds = []
    for p in preds:
        p = p.permute(0, 2, 3, 1)         # 转置
        p = torch.flatten(p, start_dim=1) # 展平
        list_preds.append(p)
    return torch.cat(list_preds, dim=1)   # 拼接

# 分类预测器
# return(B,C,H,W)
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs,                       # 输入通道数
                     num_anchors * (num_classes + 1),  # 输出通道数：每个锚框预测 num_classes+1 个值
                     kernel_size=3,                    # 卷积核大小
                     padding=1)                        # 填充大小，保持特征图尺寸不变


# 边界框回归预测器
# return(B,C,H,W)
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs,                       # 输入通道数
                     num_anchors * 4,                  # 输出通道数：每个锚框预测 4 个值（偏移量）
                     kernel_size=3,                    # 卷积核大小
                     padding=1)                        # 填充大小，保持特征图尺寸不变


# 下采样块：卷积层+批量归一化+ReLU+最大池化
# Conv -> BN -> ReLU -> MaxPool2d
# 这个函数就是一个“特征提取器 + 压缩器”，
# 用于将高分辨率、低通道的特征图，逐步转化为低分辨率、高通道的深层语义特征。
def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)



# 基本网络块：三次 down_sample_blk
def base_blk():
    blk = []
    blk.append(down_sample_blk(3, 16))
    blk.append(down_sample_blk(16, 32))
    blk.append(down_sample_blk(32, 64))
    # 三次 down_sample_blk
    return nn.Sequential(*blk)


class TinySSD(nn.Module):
    def __init__(self, sizes, ratios, classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)

        self.sizes = sizes
        self.ratios = ratios
        self.classes = classes
        self.num_classes = len(classes)
        self.num_anchors = []
        for i, (size, ratio) in enumerate(zip(sizes, ratios)):
            print("size:", size, "ratio:", ratio)
            self.num_anchors.append(len(size) + len(ratio) - 1)



        self.blk0 = base_blk()
        self.cls0 = cls_predictor(64, self.num_anchors[0], self.num_classes)
        self.bbox0 = bbox_predictor(64, self.num_anchors[0])

        self.blk1 = down_sample_blk(64, 128)
        self.cls1 = cls_predictor(128, self.num_anchors[1], self.num_classes)
        self.bbox1 = bbox_predictor(128, self.num_anchors[1])
        
        self.blk2 = down_sample_blk(128, 128)
        self.cls2 = cls_predictor(128, self.num_anchors[2], self.num_classes)
        self.bbox2 = bbox_predictor(128, self.num_anchors[2])
        
        self.blk3 = down_sample_blk(128, 128)
        self.cls3 = cls_predictor(128, self.num_anchors[3], self.num_classes)
        self.bbox3 = bbox_predictor(128, self.num_anchors[3])
        
        self.blk4 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.cls4 = cls_predictor(128, self.num_anchors[4], self.num_classes)
        self.bbox4 = bbox_predictor(128, self.num_anchors[4])




    def forward(self, X):

        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        X = self.blk0(X)
        anchors[0] = generate_anchors(X, self.sizes[0], self.ratios[0])
        cls_preds[0] = self.cls0(X)
        bbox_preds[0] = self.bbox0(X)

        X = self.blk1(X)
        anchors[1] = generate_anchors(X, self.sizes[1], self.ratios[1])
        cls_preds[1] = self.cls1(X)
        bbox_preds[1] = self.bbox1(X)

        X = self.blk2(X)
        anchors[2] = generate_anchors(X, self.sizes[2], self.ratios[2])
        cls_preds[2] = self.cls2(X)
        bbox_preds[2] = self.bbox2(X)

        X = self.blk3(X)
        anchors[3] = generate_anchors(X, self.sizes[3], self.ratios[3])
        cls_preds[3] = self.cls3(X)
        bbox_preds[3] = self.bbox3(X)

        X = self.blk4(X)
        anchors[4] = generate_anchors(X, self.sizes[4], self.ratios[4])
        cls_preds[4] = self.cls4(X)
        bbox_preds[4] = self.bbox4(X)

        # 拼接
        
        return (anchors, cls_preds, bbox_preds)



sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5]]


def display_model(model):
    print(model)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")

model = TinySSD(sizes, ratios, classes=['ball'])



