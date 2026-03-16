# Single Shot MultiBox Detector
import torch
from torch import nn

# 宽高减半块：通道数改变，宽、高减半
# down_sample_blk 是一个标准的“双卷积 + 下采样”模块
def down_sample_blk(in_channels, out_channels):
    blk = []
    # 双卷积层：不改变特征图的大小，只改变通道数
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    # 下采样层：将特征图的大小减半
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)

# 基本网络块：三次 down_sample_blk
def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    # 三次 down_sample_blk
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)



# 类别预测层：每个锚框预测num_classes+1个类别（包括背景）
# 对特征图中每个位置进行类别预测
# return(B,C,H,W)
# C = num_anchors * (num_classes + 1)
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
# 边界框预测层：每个锚框预测4个偏移量
# return(B,C,H,W)
# C = num_anchors * 4
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

# 展平预测
# pred(B,C,H,W) -> pred(B,H,W,C) -> pred(B,H*W*C)
def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)

# 连接多尺寸预测
# return(B,(H*W+...)*C)
# C = num_anchors * (num_classes + 1)
# C = num_anchors * 4
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


# 5个块网络
def get_blk(i):
    if i == 0:
        blk = base_net()                        # 三次down_sample_blk:通道数分别为3,16,32,64,宽高变小
    elif i == 1:
        blk = down_sample_blk(64, 128)          # 一次down_sample_blk:通道数翻倍，宽、高减半
    elif i == 2:
        blk = down_sample_blk(128, 128)         # 一次down_sample_blk:通道数不变，宽、高减半
    elif i == 3:
        blk = down_sample_blk(128, 128)         # 一次down_sample_blk:通道数不变，宽、高减半
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))       # 全局最大池化层：不改变通道数，将特征图的高度和宽度都压缩为1，确保不管前面层输出的特征图尺寸如何，最终都能得到统一的1×1输出
    return blk


# tiny_ssd网络
def blk_forward(X, blk, sizes, ratios, cls_predictor, bbox_predictor):
    # 1.blk:对上一层进行卷积，得到新的特征图
    # 基础网络块/多尺度特征块
    # X(B,C,H,W) -> Y(B,C,H,W)
    Y = blk(X)
    # 2.anchors:基于新的特征图生成锚框
    # 生成锚框
    # anchors(1,bpp*H*W,4)：生成的锚框
    # anchors 是预先定义的候选框，它们不依赖于具体的输入图像内容，
    # 只依赖于特征图的尺寸和预设的尺度/比例
    # 相当于一个锚框生成器
    # 尽管 anchors 的第一个维度是1，但在后续的处理中它会被广播应用到批次中的每个样本
    anchors = d2l.multibox_prior(Y, sizes, ratios)
    # print(anchors.shape)
    # torch.Size([1, 4096, 4]) # 第一层特征图：32x32,共生成4096个锚框
    # torch.Size([1, 1024, 4]) # 第二层特征图：16x16,共生成1024个锚框
    # torch.Size([1, 256, 4]) # 第三层特征图：8x8,共生成256个锚框
    # torch.Size([1, 64, 4]) # 第四层特征图：4x4,共生成64个锚框
    # torch.Size([1, 4, 4]) # 第五层特征图：1x1,共生成4个锚框
    # 3.cls:基于新的特征图生成类别预测
    # 类别预测      
    # cls_preds(B,C,H,W)，C = num_anchors * (num_classes + 1)
    # 早期的特征图的卷积层学会检测小目标，后期的特征图的卷积层学会检测大目标
    # 检测结果存储到8个通道中，每个通道对应一个锚框的类别预测
    # 后期再通过反向传播更新参数
    cls_preds = cls_predictor(Y)
    # print(cls_preds.shape)
    # 每个位置的类别预测信息需要8个数据来存储
    # torch.Size([32, 8, 32, 32]) # 第一层特征图：32x32
    # torch.Size([32, 8, 16, 16]) # 第二层特征图：16x16
    # torch.Size([32, 8, 8, 8]) # 第三层特征图：8x8
    # torch.Size([32, 8, 4, 4]) # 第四层特征图：4x4
    # torch.Size([32, 8, 1, 1]) # 第五层特征图：1x1
    # 4.bbox:基于新的特征图生成边界框预测
    # 边界框预测
    # bbox_preds(B,C,H,W)，C = num_anchors * 4
    # 早期的特征图的卷积层学会检测小目标，后期的特征图的卷积层学会检测大目标
    # 检测结果存储到16个通道中，每个通道对应一个锚框的边界框预测
    # 后期再通过反向传播更新参数
    bbox_preds = bbox_predictor(Y)
    # print(bbox_preds.shape)
    # 每个位置的边界框预测信息需要16个数据来存储
    # torch.Size([32, 16, 32, 32]) # 第一层特征图：32x32
    # torch.Size([32, 16, 16, 16]) # 第二层特征图：16x16
    # torch.Size([32, 16, 8, 8]) # 第三层特征图：8x8
    # torch.Size([32, 16, 4, 4]) # 第四层特征图：4x4
    # torch.Size([32, 16, 1, 1]) # 第五层特征图：1x1

    # Y：处理后的特征图
    # anchors(1,bpp*H*W,4)：生成的锚框
    # cls_preds(B,C,H,W)：每个锚框的类别预测
    # bbox_preds(B,C,H,W)：每个锚框的边界框预测
    return (Y, anchors, cls_preds, bbox_preds)


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5]]
num_anchors = len(sizes[0]) + len(ratios[0]) - 1 # 2 + 3 - 1 = 4


'''
目标检测算法与图片画面信息的关联是通过以下方式实现的：
1. 空间映射 ：特征图上的每个位置都对应原图的一个区域
2. 锚框机制 ：在特征图每个位置生成多个锚框，覆盖不同尺度和比例
3. 端到端学习 ：网络直接从图像像素学习到物体类别和位置的映射关系
4. 坐标变换 ：通过数学变换将特征图坐标、锚框参数与原图坐标建立联系
这种设计使得网络能够理解图像内容，并准确地定位和识别图像中的物体。
'''
class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)

        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 动态地给对象设置属性
            # 虽然 setattr 本身是 Python 内置函数
            # 但在 PyTorch 的 nn.Module 子类中使用时
            # 会触发 nn.Module 自定义的 __setattr__ 方法。
            # 特征提取器
            setattr(self, f'blk_{i}', get_blk(i))
            # 分类预测器
            setattr(self, f'cls_{i}', cls_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            # 边界框预测器
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        # print(X.shape)
        # 1.执行网络(blk,cls,bbox)
        for i in range(5):
            # X：处理后的特征图
            # anchors[i](1,bpp*H*W,4)：生成的锚框
            # cls_preds[i](B,C,H,W)：每个锚框的类别预测
            # bbox_preds[i](B,C,H,W)：每个锚框的边界框预测
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(X, getattr(self, f'blk_{i}'), sizes[i], ratios[i], getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}'))
            # print(X.shape)
            
        # 2.执行拼接
        # anchors(1,(H*W+...)*num_anchors,4)：生成的锚框
        anchors = torch.cat(anchors, dim=1)
        # cls_preds(B,(H*W+...)*num_anchors*(num_classes+1))
        cls_preds = concat_preds(cls_preds)
        # cls_preds(B,(H*W+...)*num_anchors,num_classes+1)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1)


        # bbox_preds(B,(H*W+...)*num_anchors*4)
        bbox_preds = concat_preds(bbox_preds)
        # anchors   (1,(H*W+...)*num_anchors,4)
        # cls_preds (B,(H*W+...)*num_anchors,num_classes+1)
        # bbox_preds(B,(H*W+...)*num_anchors*4)
        return anchors, cls_preds, bbox_preds   



'''
TinySSD(
  (blk_0): Sequential(
    (0): Sequential(
      (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): Sequential(
      (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (2): Sequential(
      (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU()
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
  )
  (cls_0): Conv2d(64, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bbox_0): Conv2d(64, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (blk_1): Sequential(
    (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cls_1): Conv2d(128, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bbox_1): Conv2d(128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (blk_2): Sequential(
    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cls_2): Conv2d(128, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bbox_2): Conv2d(128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (blk_3): Sequential(
    (0): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU()
    (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU()
    (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (cls_3): Conv2d(128, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bbox_3): Conv2d(128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (blk_4): AdaptiveMaxPool2d(output_size=(1, 1))
  (cls_4): Conv2d(128, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bbox_4): Conv2d(128, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
)
'''

'''
torch.Size([32, 3, 256, 256])
torch.Size([32, 64, 32, 32])
torch.Size([32, 128, 16, 16])
torch.Size([32, 128, 8, 8])
torch.Size([32, 128, 4, 4])
torch.Size([32, 128, 1, 1])
-----
output anchors: torch.Size([1, 5444, 4])
output class preds: torch.Size([32, 5444, 2])
output bbox preds: torch.Size([32, 21776])
'''


'''
第5轮执行详解
这是一个非常好的问题！我们来详细分析 TinySSD 中最后一层（blk_4 = nn.AdaptiveMaxPool2d((1, 1))）之后，cls_4 和 bbox_4 是如何工作的。

🧠 背景知识回顾
nn.AdaptiveMaxPool2d((1, 1)) 的作用是：无论输入特征图的尺寸是多少，都会将其压缩为 1×1 的空间尺寸，但保留通道数不变。
在 TinySSD 中，blk_4 的输入通道数是 128（来自 blk_3 的输出），所以：
输入到 blk_4 的张量形状可能是 (B, 128, H, W)（比如 (B, 128, 3, 3) 或 (B, 128, 2, 2) 等）
输出后变成 (B, 128, 1, 1)

然后，这个 (B, 128, 1, 1) 的特征图会传给：
cls_4 = Conv2d(128, num_anchors (num_classes + 1), kernel_size=3, padding=1)
bbox_4 = Conv2d(128, num_anchors 4, kernel_size=3, padding=1)

❓问题核心
当特征图已经是 1×1 时，用 kernel_size=3, padding=1 的卷积还能正常工作吗？
✅ 答案：可以，但需要理解 PyTorch 的行为

PyTorch 的 Conv2d 在这种情况下 不会报错，因为 padding=1 会让输入在空间维度上“虚拟扩展”：
原始输入：(B, 128, 1, 1)
应用 padding=1 后，等效于 (B, 128, 3, 3)（四周补0）
然后用 3×3 卷积核滑动，只有一个有效输出位置（中心点）
所以输出仍然是 (B, C_out, 1, 1)

✅ 结论：虽然看起来“不合理”，但数学上和实现上都是合法的，输出尺寸保持 1×1。

🔍 具体计算（以 cls_4 为例）

假设：
num_classes = 20（如 VOC 数据集）
num_anchors = 4（如你代码中定义的）
那么 cls_4 的输出通道数 = 4 × (20 + 1) = 84
bbox_4 的输出通道数 = 4 × 4 = 16

输入：X = (B, 128, 1, 1)

执行：
python
cls_pred = cls_4(X) # Conv2d(128, 84, 3, padding=1)

内部过程：
padding 后：(B, 128, 3, 3)
卷积后：每个 3×3 区域 → 一个标量，共 1×1 个位置
输出：(B, 84, 1, 1)

同理，bbox_4(X) → (B, 16, 1, 1)

📌 锚框生成：d2l.multibox_prior(Y, sizes, ratios)
Y 的 shape 是 (B, 128, 1, 1)
multibox_prior 只关心 Y 的 空间尺寸（H=1, W=1）
它会在 1×1 的网格上 为每个位置生成 num_anchors = 4 个锚框
所以总共生成 1×1×4 = 4 个锚框
输出 anchors shape: (1, 4, 4)（格式：[center_x, center_y, width, height]）

这些锚框对应的是整张原图的非常大尺度的目标（因为这是最深层、感受野最大的特征图）。

🧩 整体意义
SSD/TinySSD 使用 多尺度特征图 进行检测：
浅层（高分辨率）→ 检测小目标
深层（低分辨率，如 1×1）→ 检测大目标（甚至覆盖全图）
blk_4 的 1×1 特征图虽然空间信息极少，但语义信息最强（高层语义）
用它预测 极少数（4个）大尺度锚框 是合理的设计

⚠️ 潜在注意点

虽然代码能运行，但有些细节值得思考：

1. kernel_size=3 在 1×1 上是否必要？
实际上，此时 kernel_size=1 就足够了（因为感受野已全覆盖）
使用 3×3 只是保持结构统一，无实质增益，但也不影响结果

2. 如果输入图像太小（如 < 32×32），前面层可能提前变成 1×1
会导致多个阶段的特征图都是 1×1，锚框重复
所以 SSD 通常要求输入图像 ≥ 300×300

✅ 总结

步骤 输入 Shape 操作 输出 Shape
------ ----------- ------ ------------
blk_4 (B, 128, H, W) AdaptiveMaxPool2d((1,1)) (B, 128, 1, 1)
cls_4 (B, 128, 1, 1) Conv2d(128, 84, 3, padding=1) (B, 84, 1, 1)
bbox_4 (B, 128, 1, 1) Conv2d(128, 16, 3, padding=1) (B, 16, 1, 1)
multibox_prior 基于 (1,1) 网格 生成 4 个锚框 (1, 4, 4)
所以：尽管特征图是 1×1，cls_4 和 bbox_4 依然能正常工作，输出对应 4 个大尺度锚框的预测。这是 SSD 多尺度检测的关键一环。

如有需要，我也可以帮你可视化这些锚框在原图上的位置 😊
'''