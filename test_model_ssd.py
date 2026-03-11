import torch
from torch import nn
import torchvision.io as tv_io
from common import generate_anchors
from common import load_data_bananas
from common import anchor_to_label






# 分类预测器
# 输入(B,C_in,H,W)
# 输出(B,C_out,H,W)
def cls_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(num_inputs,                       # 输入通道数
                     num_anchors * (num_classes + 1),  # 输出通道数：每个锚框预测 num_classes+1 个值
                     kernel_size=3,                    # 卷积核大小
                     padding=1)                        # 填充大小，保持特征图尺寸不变


# 边界框回归预测器
# 输入(B,C_in,H,W)
# 输出(B,C_out,H,W)
def bbox_predictor(num_inputs, num_anchors):
    return nn.Conv2d(num_inputs,                       # 输入通道数
                     num_anchors * 4,                  # 输出通道数：每个锚框预测 4 个值（偏移量）
                     kernel_size=3,                    # 卷积核大小
                     padding=1)                        # 填充大小，保持特征图尺寸不变

# 拼接多尺寸预测结果
# return(B,(H*W+...)*C_out)
def concat_preds(preds):
    list_preds = []
    for p in preds:
        p = p.permute(0, 2, 3, 1)         # 转置(B,H,W,C_out)
        p = torch.flatten(p, start_dim=1) # 展平(B,H*W*C_out)
        list_preds.append(p)
    return torch.cat(list_preds, dim=1)   # 拼接(B,(H*W+...)*C_out)

# 下采样块：卷积层+批量归一化+ReLU+最大池化
# 双卷积：Conv -> BN -> ReLU 
# 下采样：MaxPool2d
# 这个函数就是一个“特征提取器 + 压缩器”
# 用于将高分辨率、低通道的特征图，逐步转化为低分辨率、高通道的深层语义特征。
# 输入(N,C_in,H,W)
# 输出(N,C_out,H/2,W/2)
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
# 分辨率在下降，通道数在上升
def base_blk():
    blk = []
    blk.append(down_sample_blk(3, 16))
    blk.append(down_sample_blk(16, 32))
    blk.append(down_sample_blk(32, 64))
    # 三次 down_sample_blk
    return nn.Sequential(*blk)


def display_model(model):
    print(model)
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params}")


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



        self.blk0 = base_blk() # 三次 down_sample_blk
        self.cls0 = cls_predictor(64, self.num_anchors[0], self.num_classes)
        self.bbox0 = bbox_predictor(64, self.num_anchors[0])

        self.blk1 = down_sample_blk(64, 128) # 一次 down_sample_blk
        self.cls1 = cls_predictor(128, self.num_anchors[1], self.num_classes)
        self.bbox1 = bbox_predictor(128, self.num_anchors[1])
        
        self.blk2 = down_sample_blk(128, 128) # 一次 down_sample_blk
        self.cls2 = cls_predictor(128, self.num_anchors[2], self.num_classes)
        self.bbox2 = bbox_predictor(128, self.num_anchors[2])
        
        self.blk3 = down_sample_blk(128, 128) # 一次 down_sample_blk
        self.cls3 = cls_predictor(128, self.num_anchors[3], self.num_classes)
        self.bbox3 = bbox_predictor(128, self.num_anchors[3])
        
        self.blk4 = nn.AdaptiveAvgPool2d(output_size=(1, 1)) # 平均池化层：(B,C,H,W)->(B,C,1,1)
        self.cls4 = cls_predictor(128, self.num_anchors[4], self.num_classes)
        self.bbox4 = bbox_predictor(128, self.num_anchors[4])


    def forward(self, X):

        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5

        X = self.blk0(X) # 生成特征图
        anchors[0] = generate_anchors(X, self.sizes[0], self.ratios[0])
        cls_preds[0] = self.cls0(X)
        bbox_preds[0] = self.bbox0(X)

        X = self.blk1(X) # 继续生成特征图
        anchors[1] = generate_anchors(X, self.sizes[1], self.ratios[1])
        cls_preds[1] = self.cls1(X)
        bbox_preds[1] = self.bbox1(X)

        X = self.blk2(X) # 继续生成特征图
        anchors[2] = generate_anchors(X, self.sizes[2], self.ratios[2])
        cls_preds[2] = self.cls2(X)
        bbox_preds[2] = self.bbox2(X)

        X = self.blk3(X) # 继续生成特征图
        anchors[3] = generate_anchors(X, self.sizes[3], self.ratios[3])
        cls_preds[3] = self.cls3(X)
        bbox_preds[3] = self.bbox3(X)

        X = self.blk4(X) # 继续生成特征图
        anchors[4] = generate_anchors(X, self.sizes[4], self.ratios[4])
        cls_preds[4] = self.cls4(X)
        bbox_preds[4] = self.bbox4(X)

        # 拼接多尺度层的锚框、分类、边框偏移量
        # (1,N(H*W+...),4)
        batch_anchors = torch.cat(anchors, dim=1)
        # (B,(H*W+...)*C_out)
        batch_pred_classes = concat_preds(cls_preds)
        # (B,(H*W+...)*C_out)
        batch_pred_offset = concat_preds(bbox_preds)

        # (1,5380,4)
        # (32,10760)
        # (32,21520)
        # 把类别数也返回
        return (batch_anchors, batch_pred_classes, batch_pred_offset, self.num_classes)


def train_tinyssd(net, train_iter, device, num_epochs=20):
    # 定义损失函数
    cls_loss = nn.CrossEntropyLoss(reduction='none') # 分类损失：交叉熵损失：预测值与标签值的负对数似然损失
    bbox_loss = nn.L1Loss(reduction='none')          # 回归损失：L1范数损失：预测值与标签值的差的绝对值，L1Loss(x,y)=∣x−y∣
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

    # 将模型移动到设备
    net = net.to(device)


    # batch_pred_classes:模型生成的类别
    # batch_pred_offset:模型生成的边界框偏移量

    # batch_assigned_classes:分配的类别标签
    # batch_assigned_offset:分配的偏移量标签
    # batch_assigned_mask:分配的掩码标签
    # 返回损失值
    def calc_loss(batch_pred_classes, batch_pred_offset, batch_assigned_classes, batch_assigned_offset, batch_assigned_mask, num_classes):
        # batch_pred_classes (B,(H*W+...)*C_out)
        batch_size = batch_pred_classes.shape[0]
        num_classes = num_classes + 1
        cls = cls_loss(batch_pred_classes.reshape(-1, num_classes), batch_assigned_classes.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        bbox = bbox_loss(batch_pred_offset * batch_assigned_mask, batch_assigned_offset * batch_assigned_mask).mean(dim=1)
        # 而边界框回归只对正样本有意义（只有正样本才有真实的边界框偏移量）
        # (B)
        return cls + bbox
    
    for i in range(num_epochs):
        metric = Accumulator(4)
        net.train()
        for image, target in train_iter:
            optimizer.zero_grad()
            image = image.to(device) # (B,C,H,W)
            target = target.to(device) # (B,1,5)
            # 模型算出锚框、分类、边框偏移量
            batch_anchors, batch_pred_classes, batch_pred_offset, num_classes = net(image)
            # 使用锚框和数据集标签算出类别、偏移量、掩码
            # 其实这步就是目标检测和图像分类的区别所在
            batch_assigned_classes, batch_assigned_offset, batch_assigned_mask = anchor_to_label(batch_anchors, target)
            # 计算损失函数(分类损失+回归损失)
            l = calc_loss(batch_pred_classes, batch_pred_offset, batch_assigned_classes, batch_assigned_offset, batch_assigned_mask, num_classes)
            # 反向传播和优化
            l.mean().backward()
            # 更新参数
            optimizer.step()

        
            # 切断梯度计算
            with torch.no_grad():
                # 计算分类准确率和边界框回归准确率
                metric.add(eval_class(batch_pred_classes, batch_assigned_classes, num_classes), batch_assigned_classes.numel(), eval_offset(batch_pred_offset, batch_assigned_offset, batch_assigned_mask), batch_assigned_offset.numel())
                print(f"batch loss: {l.mean().item():.6f}")
        
        class_err = 1 - metric[0] / metric[1]
        bbox_mae = metric[2] / metric[3]
        print(f"epoch {i + 1}:" f"class err: {class_err:.6f}, bbox mae: {bbox_mae:.6f}")

# 评估分类准确率
def eval_class(batch_pred_classes,batch_assigned_classes,num_classes):
    # (B,num_anchors*num_classes+1)->(B,num_anchors,num_classes+1)
    batch_pred_classes = batch_pred_classes.reshape(batch_pred_classes.shape[0], -1, num_classes + 1)
    # (B,num_anchors,num_classes+1)->(B,num_anchors)
    batch_pred_classes = batch_pred_classes.argmax(dim=-1)
    return float((batch_pred_classes.type(batch_assigned_classes.dtype) == batch_assigned_classes).sum())
# 评估offset准确率
def eval_offset(batch_pred_offset,batch_assigned_offset,batch_assigned_mask):
    return float((torch.abs((batch_pred_offset - batch_assigned_offset) * batch_assigned_mask)).sum())

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




def main():
    # 设置打印选项，保留两位小数
    torch.set_printoptions(precision=2) 
    # 设置超参数
    num_epochs = 20
    batch_size = 32
    # 获取可用设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    # 加载数据集
    batch_size = 32
    train_iter, valid_iter = load_data_bananas(batch_size)
    # feature: torch.Size([32, 3, 256, 256])
    # label: torch.Size([32, 1, 5])



    # 定义模型
    sizes = [[0.2, 0.272], [0.37, 0], [0.], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5], [1, 2, 0.5]]
    classes = ['banana']
    net = TinySSD(sizes, ratios, classes)
    # 打印模型结构
    # display_model(net)

    # 加载模型（若存在则加载，否则从头训练）
    try:
        net.load_state_dict(torch.load('./pth/tiny_ssd_model.pth', map_location=device))
        print('已加载预训练模型 ./pth/tiny_ssd_model.pth')
        # 预测
        # return 
    except FileNotFoundError:
        print('未找到预训练模型，将从头开始训练')


    # 训练模型
    train_tinyssd(net, train_iter, device, num_epochs)

    # 保存模型
    torch.save(net.state_dict(), './pth/tiny_ssd_model.pth')
    print('模型已保存至 ./pth/tiny_ssd_model.pth')



    # # 评估模型
    # print('-----')
    # print('评估模型')
    # X = tv_io.read_image('./img/banana.jpg').unsqueeze(0).float()
    # print(X.shape)
    # # ouput(筛选出的锚框总数,6):6(class_id, scores, predicted_bb)
    # output = predict_tinyssd(net, X, device)
    # print(f'筛选出的锚框：{output.shape}')
    
    # print(output)

    # # 可视化结果
    # img = X.squeeze(0).permute(1, 2, 0).long()
    # display(img, output, threshold=0.9)


if __name__ == '__main__':
    main()



