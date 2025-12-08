import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from common import TinySSD
from common import multibox_detection
from common import display
import matplotlib.pyplot as plt

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
# cls_preds  (B,(H*W+...)*num_anchors,num_classes+1)：每个锚框的类别概率
# cls_labels (B,(H*W+...)*num_anchors)：每个锚框的真实类别标签
def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维。
    # 分类正确的样本总数（不是准确率，而是正确个数）
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())
# 评估函数：边界框
# bbox_preds  (B,(H*W+...)*num_anchors*4)：每个锚框的预测边界框坐标
# bbox_labels (B,(H*W+...)*num_anchors*4)：每个锚框的真实边界框坐标
# bbox_masks  (B,(H*W+...)*num_anchors*4)：每个锚框的有效掩码（1表示有效，0表示无效）
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
            # 就近匹配：每个锚框分配一个真实边界框，计算偏移量和掩码，用于训练
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

'''
功能 :
- 将预测的边界框偏移量应用到锚框上，得到实际的预测框
- 使用非极大值抑制(NMS)去除重复的检测框
- 根据置信度阈值过滤低质量的预测结果
- 返回最终的检测结果，包括类别、置信度和边界框坐标

输出 :
  - output : 形状为 (1, NAC, 6) 的张量，其中每一行包含：
  - 第0列:类别ID(-1表示背景)
  - 第1列:置信度分数
  - 第2-5列:边界框坐标(x1, y1, x2, y2)
'''
# X(torch.Size([B=1, 3, 256, 256]))
def predict_tinyssd(net, X, device):
    net.eval()
    # anchors   (1,(H*W+...)*num_anchors,4)
    # cls_preds (B,(H*W+...)*num_anchors,num_classes+1)：包括背景0的原始得分
    # bbox_preds(B,(H*W+...)*num_anchors*4)
    # cls_preds:卷积层输出的原始得分，存储在8个通道中，每个通道对应一个锚框的类别预测
    # bbox_preds:卷积层输出的边界框预测，存储在16个通道中，每个通道对应一个锚框的边界框预测
    with torch.no_grad():
        anchors, cls_preds, bbox_preds = net(X.to(device))
        print('-----')
        print(anchors.shape)
        print(cls_preds.shape)
        print(bbox_preds.shape)
        print('-----')
        # cls_probs(B,num_classes+1,(H*W+...)*num_anchors):(B,类别数,锚框数)
        # cls_preds是原始得分
        # cls_probs是归一化后的概率：包括背景0的类别概率
        cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
        print(cls_probs.shape)
        # cls_probs(B,num_classes+1,(H*W+...)*num_anchors):(B,类别数,锚框数)
        # bbox_preds(B,(H*W+...)*num_anchors*4)
        # anchors(1,(H*W+...)*num_anchors,4)
        # output(B,(H*W+...)*num_anchors,6):6(class_id, scores, predicted_bb)
        # cls_probs是归一化后的概率：包括背景0的类别概率
        output = multibox_detection(cls_probs, bbox_preds, anchors)
        # B=1
        # output(B,(H*W+...)*num_anchors,6):6(class_id, scores, predicted_bb)
        # output[0]:class_id(锚框总数,)
        # idx:非背景锚框的索引列表
        idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
        # 假设 output[0] 如下所示：
        # (class_id, score, x1, y1, x2, y2)
        # [
        #     [-1, 0.95, 0.1, 0.1, 0.3, 0.3],    # 背景锚框
        #     [0,  0.85, 0.2, 0.2, 0.4, 0.4],    # 类别0的锚框（如"狗"）
        #     [-1, 0.92, 0.5, 0.5, 0.7, 0.7],    # 背景锚框
        #     [1,  0.90, 0.6, 0.6, 0.8, 0.8]     # 类别1的锚框（如"猫"）
        # ]
        # 那么：
        # idx = [1, 3] （索引1和3对应的class_id不为-1）
        # output[0, idx] 结果为：
        # [
        #     [0,  0.85, 0.2, 0.2, 0.4, 0.4],    # 类别0的锚框
        #     [1,  0.90, 0.6, 0.6, 0.8, 0.8]     # 类别1的锚框
        # ]   
        # output[0, idx] 表示从模型预测结果中提取出的第一个样本的所有有效（非背景）检测结果
        # 这是pytorch的语法
        return output[0, idx]

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
    print('-----')
    print('评估模型')
    X = torchvision.io.read_image('./img/banana.jpg').unsqueeze(0).float()
    print(X.shape)
    # ouput(筛选出的锚框总数,6):6(class_id, scores, predicted_bb)
    output = predict_tinyssd(net, X, device)
    print(f'筛选出的锚框：{output.shape}')
    
    print(output)

    # 可视化结果
    img = X.squeeze(0).permute(1, 2, 0).long()
    display(img, output, threshold=0.9)


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