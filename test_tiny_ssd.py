import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from common import TinySSD
from common import display_model

# anchors   (1,(H*W+...)*num_anchors,4):锚框生成器
# labels: (B,1,5)
def multibox_target(anchors, labels):
    device = anchors.device
    batch_offset, batch_mask, batch_class_labels = [], [], []
    # anchors: (NAC,4):相当于一个网格模板，每个样本的锚框都是使用这个模板
    anchors = anchors.squeeze(0)
    num_anchors = anchors.shape[0]
    batch_size = labels.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        # anchors_bbox_map(NAC,):-1表示anchor未bind到gt,anchor是负样本
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        # offset: (NAC,4)
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        # offset: (NAC,4)->(NAC*4)
        batch_offset.append(offset.reshape(-1))
        # bbox_mask: (NAC,4)->(NAC*4)
        batch_mask.append(bbox_mask.reshape(-1))
        # class_labels: (NAC,)
        batch_class_labels.append(class_labels)
    # bbox_offset: (B,NAC*4)
    # 锚框到真实边界框的偏移量
    bbox_offset = torch.stack(batch_offset)
    # bbox_mask: (B,NAC*4)
    bbox_mask = torch.stack(batch_mask)
    # class_labels: (B,NAC) 
    # 锚框对应的真实边界框的类别标签
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)



# 评估函数：类别
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    # 分类正确的样本总数（不是准确率，而是正确个数）
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())
# 评估函数：边界框
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    # 所有有效边界框的 L1 误差总和（越小越好）
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

def train_tinyssd(net, train_iter, device, num_epochs=20):
    # 定义损失函数
    cls_loss = nn.CrossEntropyLoss(reduction='none') # 分类损失：交叉熵损失：预测值与标签值的负对数似然损失
    bbox_loss = nn.L1Loss(reduction='none')          # 回归损失：L1范数损失：预测值与标签值的差的绝对值，L1Loss(x,y)=∣x−y∣
    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)



    # cls_preds (B,(H*W+...)*num_anchors,num_classes+1)
    # bbox_preds(B,(H*W+...)*num_anchors*4)
    def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
        # cls_preds (B,(H*W+...)*num_anchors,num_classes+1)
        batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
        # B*(H*W+...)*num_anchors->(B,(H*W+...)*num_anchors)->(B)
        # 对于批次中的每个样本，计算其所有锚框的分类损失的平均值
        cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
        # (B,(H*W+...)*num_anchors*4)->(B)
        # 对于批次中的每个样本，计算其所有锚框的回归损失的平均值
        bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
        # (B)
        return cls + bbox

    timer = d2l.Timer()
    net = net.to(device)

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(4)
        net.train()
        # feature: torch.Size([32, 3, 256, 256])
        # label: torch.Size([32, 1, 5])
        for features, target in train_iter:
            timer.start()
            optimizer.zero_grad()
            # X: torch.Size([32, 3, 256, 256])
            # Y: torch.Size([32, 1, 5])
            X, Y = features.to(device), target.to(device)
            # 生成多尺度的锚框，为每个锚框预测类别和偏移量
            # anchors   (1,(H*W+...)*num_anchors,4):锚框生成器
            # cls_preds (B,(H*W+...)*num_anchors,num_classes+1)
            # bbox_preds(B,(H*W+...)*num_anchors*4)
            # features:分类、边框偏移量
            anchors, cls_preds, bbox_preds = net(X)
            # 为每个锚框标注类别和偏移量
            # anchors   (1,(H*W+...)*num_anchors,4)
            # Y         (B,1,5)
            # bbox_labels(B,(H*W+...)*num_anchors*4)
            # bbox_masks(B,(H*W+...)*num_anchors*4)
            # cls_labels(B,(H*W+...)*num_anchors)
            # labels：分类，边框偏移量
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
            # 根据类别和偏移量的预测和标注值计算损失函数
            l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)
            # 使用 l.mean() 计算整个批次的平均损失
            l.mean().backward()
            optimizer.step()
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(), bbox_eval(bbox_preds, bbox_labels, bbox_masks), bbox_labels.numel())
            with torch.no_grad():
                print("batch loss:", l.mean().item())
        # 这里 metric[0] / metric[1] 是分类准确率（正确预测数/总样本数）
        # 所以 1 - 准确率 就是分类错误率
        # 这里 metric[2] / metric[3] 是边界框预测的平均绝对误差（总误差/总元素数）。
        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        print(f'epoch {epoch + 1}:' f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
    print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on ' f'{str(device)}')


def main():
    # 设置超参数

    # 获取可用设备
    device = d2l.try_gpu()
    print(f'Using device: {device}')


    # 加载数据集
    batch_size = 32
    # （feature,label）
    # feature: torch.Size([32, 3, 256, 256])
    # label: torch.Size([32, 1, 5])
    train_iter, valid_iter = d2l.load_data_bananas(batch_size)


    # 定义模型
    net = TinySSD(num_classes=1)
    # 打印模型结构
    # display_model(net)


    # 训练模型
    train_tinyssd(net, train_iter, device, num_epochs=20)


    # 评估模型

if __name__ == '__main__':
    main()




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