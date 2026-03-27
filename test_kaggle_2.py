import os
import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
import shutil
import collections
import math


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


def reorg_dog_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


transform_train = torchvision.transforms.Compose([
    # 随机裁剪图像，所得图像为原始面积的0.08～1之间，高宽比在3/4和4/3之间。
    # 然后，缩放图像以创建224x224的新图像
    torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                             ratio=(3.0/4.0, 4.0/3.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    # 随机更改亮度，对比度和饱和度
    torchvision.transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4),

    torchvision.transforms.ToTensor(),
    # 标准化图像的每个通道
    # Normalize 的操作 (像素值 - 均值) / 标准差 是一个线性变换
    # 它对图像中的每一个像素都应用了完全相同的数学规则。
    # 这就好比给一张照片整体调亮或调暗，虽然每个点的亮度值都变了，但照片里物体的轮廓、明暗对比关系完全没有变。
    # transforms.Normalize 并不是在破坏信息，而是在翻译信息。
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])


transform_test = torchvision.transforms.Compose([
    # 规则是：将图像的较短边缩放到 256 像素，而较长边则按原始比例等比缩放。
    torchvision.transforms.Resize(256),
    # 从图像中心裁切224x224大小的图片
    # 这保证了最终输入模型的图像尺寸是统一的，同时也保留了图像最核心的部分。
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])



def train(net, device, train_iter, valid_iter, num_epochs,lr,wd,momentum,step_size,gamma):
    net.to(device)
    net.train()
    # 损失函数
    loss = nn.CrossEntropyLoss(reduction="none")
    # 初始化优化器(只更新可训练参数)
    optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr, momentum=momentum, weight_decay=wd)
    # 初始化学习率衰减器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    # 保存每个epoch的损失
    epoch_losses = []
    for epoch in range(num_epochs):
        for i, (featrues,labels) in enumerate(train_iter):
            featrues = featrues.to(device)
            labels = labels.to(device)
            # 清空梯度
            net.zero_grad()
            # 前向传播
            pred = net(featrues)
            # 计算损失(1个batch的平均损失)
            l = loss(pred, labels).mean()
            # 反向传播
            l.backward()
            # 更新参数
            optimizer.step()
            with torch.no_grad():
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {l.item():.4f}")

        # 学习率衰减
        scheduler.step()
        # 保存当前epoch的最后一个batch的损失
        epoch_losses.append(l.item())
    
    # 绘制损失折线图
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./table/epoch_loss_curve.png')
    plt.show()
    

def evaluate(net,device,valid_iter):
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

    pass

def main():
    # 定义超参数 
    num_classes = 120 # 120个狗品种
    batch_size = 32
    num_epochs = 20
    data_dir = './data/kaggle_dog_tiny'
    lr = 1e-3
    wd = 1e-4
    momentum=0.9
    step_size = 2
    gamma = 0.9
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_ds, train_valid_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),transform=transform_train) for folder in ['train', 'train_valid']]
    valid_ds, test_ds = [torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train_valid_test', folder),transform=transform_test) for folder in ['valid', 'test']]
    train_iter, train_valid_iter = [torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)for dataset in (train_ds, train_valid_ds)]
    valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)

    # 定义网络
    # 创建一个空的、顺序执行的神经网络容器
    finetune_net = nn.Sequential()
    # ResNet-34模型输出1000个类别
    finetune_net.features = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    # 定义一个新的输出网络，共有num_classes个输出类别
    finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256), nn.ReLU(),nn.Linear(256, num_classes))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(device)
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False



    # 加载网络
    if os.path.exists('./pth/kaggle_resnet34.pth'):
        finetune_net.load_state_dict(torch.load('./pth/kaggle_resnet34.pth'))
        print('模型已加载')
    else:
        print('模型不存在')
        # 训练模型
        train(finetune_net, device, train_iter, valid_iter, num_epochs,lr,wd,momentum,step_size,gamma)
        # 保存模型
        torch.save(finetune_net.state_dict(), './pth/kaggle_resnet34.pth')

    
    # 评估模型
    evaluate(finetune_net, device, valid_iter)


   


if __name__ == '__main__':
    main()