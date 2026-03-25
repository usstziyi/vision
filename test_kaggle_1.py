import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn



def read_csv_labels(fname):
    with open(fname, 'r') as f:
        # 跳过文件头行(列名)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))


def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    # 直接覆盖
    shutil.copy(filename, target_dir)

# 一套代码就同时为两种评估方式做好了数据准备
def reorg_train_valid(data_dir, labels, valid_ratio):
    """将验证集从原始的训练集中拆分出来"""
    # 训练数据集中样本最少的类别中的样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # 验证集中每个类别的样本数
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        # train_file 是文件全名
        # labels 是字典：{文件前名: 类别名}
        # label 是类别名
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)

        # 方式一：train_valid：用于 K折交叉验证，它代表了可用于模型训练和内部验证的全部原始数据。
        copyfile(fname, os.path.join(data_dir, 'train_valid_test','train_valid', label))

        # 方式二：train + valid：用于标准的、一次性的训练/验证划分。
        if label not in label_count or label_count[label] < n_valid_per_label:
            # 验证集
            copyfile(fname, os.path.join(data_dir, 'train_valid_test','valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            # 训练集
            copyfile(fname, os.path.join(data_dir, 'train_valid_test','train', label))
    return n_valid_per_label



def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        # 测试集
        copyfile(os.path.join(data_dir, 'test', test_file), os.path.join(data_dir, 'train_valid_test', 'test', 'unknown'))

def reorg_cifar10_data(data_dir, valid_ratio):
    # (文件名:类别名)
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    # 生成两种数据集(K折、常规训练/验证划分)
    reorg_train_valid(data_dir, labels, valid_ratio)
    # (常规测试集：位置类别)
    reorg_test(data_dir)


transform_train = torchvision.transforms.Compose([
    # 在高度和宽度上将图像放大到40像素的正方形
    torchvision.transforms.Resize(40),
    # 生成一个面积为原始图像面积0.64～1倍的小正方形，
    # 然后将其缩放为高度和宽度均为32像素的正方形
    # 然后在可行的区域随机裁剪出一个高度和宽度均为32像素的正方形图像
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 1.将 PIL 图像或 NumPy 数组转换为 PyTorch 的 Tensor 格式
    # 2.将图像的通道数从 H x W x C 转换为 C x H x W
    # 3.将图像的像素值从 [0, 255] 转换为 [0, 1]
    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])


transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

# lr: 学习率,表示每次更新参数时的步长
# step_size: 每隔多少个 epoch 进行一次学习率衰减
# gamma: 表示每次衰减时，学习率乘位原来的多少倍
def train(net, device, train_iter, valid_iter, num_epochs,lr,wd,momentum,step_size,gamma):
    net.to(device)
    net.train()
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction="none")
    # 初始化优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    # 初始化学习率衰减器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    for epoch in range(num_epochs):
        for i, (featrues,labels) in enumerate(train_iter):
            featrues = featrues.to(device)
            labels = labels.to(device)
            # 清空梯度
            net.zero_grad()
            # 前向传播
            pred = net(featrues)
            # 计算损失
            l = loss(pred, labels)
            # 反向传播
            l.sum().backward()
            # 更新参数
            optimizer.step()
        
            # 禁用梯度计算，节省内存并加速计算（仅用于打印损失，不需要反向传播）
            # 不影响下一轮训练，因为每次迭代都会重新计算梯度
            # 这里只是用于打印损失，禁用梯度计算可以节省内存
            with torch.no_grad():
                # 打印损失
                print(f'Epoch {epoch+1}, Batch {i}, Loss: {l.sum()/featrues.shape[0]:.4f}')
        # 学习率衰减
        scheduler.step()





def evaluate(net,valid_iter,device):
    net.to(device)
    net.eval()
    with torch.no_grad():
        acc = 0.0
        for featrues,labels in valid_iter:
            featrues = featrues.to(device)
            labels = labels.to(device)
            # 前向传播
            pred = net(featrues)
            # 计算准确率
            batch_acc = (pred.argmax(dim=1) == labels).sum().item()/featrues.shape[0]
            # 累加准确率
            acc += batch_acc
            print(f'batch Acc: {batch_acc:.4f}')
    print(f'Valid Acc: {acc / len(valid_iter):.4f}')





def main():
    # 定义超参数
    num_classes = 10
    batch_size = 32
    num_epochs = 100
    data_dir = './data/kaggle_cifar10_tiny'
    lr = 2e-4
    wd = 5e-4
    momentum=0.9
    step_size = 4
    gamma = 0.9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #加载数据集
    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),transform=transform_train) for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),transform=transform_test) for folder in ['valid', 'test']]
    train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True) for dataset in (train_ds, train_valid_ds)]
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)  

    # 定义网络
    from d2l import torch as d2l
    net = d2l.resnet18(num_classes,in_channels=3)


    # 判断模型是否存在
    if os.path.exists('./pth/kaggle_resnet18.pth'):
        net.load_state_dict(torch.load('./pth/kaggle_resnet18.pth'))
        print('模型已加载')
    else:
        print('模型不存在')

        # 训练模型
        train(net, device, train_iter, valid_iter, num_epochs,lr,wd,momentum,step_size,gamma)

        # 保存模型
        torch.save(net.state_dict(), './pth/kaggle_resnet18.pth')



    # 评估模型
    evaluate(net,valid_iter,device)
                                    
                                
if __name__ == '__main__':
    main()