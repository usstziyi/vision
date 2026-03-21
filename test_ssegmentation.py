import os
import torch
from torch import nn
import torchvision
import torchvision.models as models

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]




VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']




# 存
def voc_colormap2label():
    """构建从RGB到VOC类别索引的映射"""
    colormap2labelindices = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        # 等价：colormap2label[colormap[0] * 256 * 256 + colormap[1] * 256 + colormap[2]] = i
        colormap2labelindices[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i # 存
    # shape(256**2,)=(262144,)
    return colormap2labelindices # 一共21种像素值

# 取:把影子像素一一映射到类别索引
def voc_label_indices(label, colormap2labelindices):
    """将VOC标签中的RGB值映射到它们的类别索引"""
    colormap = label.permute(1, 2, 0).numpy().astype('int32')
    idx = (colormap[:, :, 0] * 256 * 256 + colormap[:, :, 1] * 256 + colormap[:, :, 2])
    # idx.shape: (320, 480)
    # torch.Size([320, 480])
    return colormap2labelindices[idx] # 取


# 随机裁剪特征和标签图像
def voc_rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

def read_voc_images(voc_dir, is_train=True):
    """读取所有VOC图像并标注"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation','train.txt' if is_train else 'val.txt')
    # PNG 格式可以是单通道的灰度图（常用于表示像素类别索引），也可以是多通道的。
    # 为了确保标签图像是单通道的整数索引，代码明确指定了 mode=ImageReadMode.RGB
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
    return features, labels

class VOCSegDataset(torch.utils.data.Dataset):
    """一个用于加载VOC数据集的自定义数据集"""

    def __init__(self, is_train, crop_size, voc_dir):
        # 图像预处理的标准化（Normalization）变换对象
        # 这个函数是专门为处理像素值在 [0, 1] 范围内的浮点型图像而设计的
        # output[channel] = (input[channel] - mean[channel]) / std[channel]
        self.transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)

        self.features = [self.normalize_image(feature) for feature in self.filter(features)]
        self.labels = self.filter(labels)
        
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        # 裁切
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # 转换为类别索引
        voc_label = voc_label_indices(label, self.colormap2label)
        # 返回特征和标签，标签的形状为(h,w)
        return (feature, voc_label)

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = "./data/VOCdevkit/VOC2012"
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir), batch_size,shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir), batch_size,drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


# 这段代码其实并不是在执行双线性插值，而是利用双线性插值的核（Kernel）思想来初始化一个卷积层的权重。
# 这个函数的目的是创建一个特殊的卷积核（也叫滤波器）。
# 这个卷积核的形状是 (in_channels, out_channels, kernel_size, kernel_size)。
# 它的值不是随机的，而是根据双线性插值的权重公式计算出来的。这样，当这个卷积核对图像进行卷积操作时，其效果就类似于双线性插值，从而实现平滑的上采样。
# 这些计算是为了让生成的权重矩阵 filt 以中心为原点，向四周扩散。
# 它是一个中心亮、四周暗的对称核，非常适合做平滑的上采样。
def bilinear_kernel(in_channels, out_channels, kernel_size):
    # 计算缩放因子，决定了权重变化的快慢
    factor = (kernel_size + 1) // 2 # 物理中点
    # 计算序列中点
    if kernel_size % 2 == 1: 
        center = factor - 1 # 序列中点
    else:
        center = factor - 0.5 # 序列中点

    # og[0](k,1):y坐标
    # og[1](1,k):x坐标
    og = (torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1))
    # 而使用 factor=3，可以使整个 5x5 的核都具有非零权重，从而产生更平滑、影响范围更大的上采样效果
    filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)

    weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
    # NumPy/PyTorch 的高级索引规则：当多个维度的索引数组长度相同时，它们会被“配对”使用
    # 在双线性插值上采样的场景下，我们希望每个输入通道的特征图被独立地放大，并直接“传递”到对应的输出通道。
    # 我们不希望输入通道 A 的信息与输入通道 B 的信息在上采样过程中混合。
    # 而像 weight[0, 1, :, :]、weight[1, 0, :, :] 等交叉项都保持为 0，确保了通道间的独立性。
    # 分组卷积（Grouped Convolution）的一种特例，常用于通道独立的上采样。
    weight[range(in_channels), range(out_channels), :, :] = filt
    
    return weight



def train(net, train_iter,test_iter,num_epochs,device):
    net.to(device)
    net.train()

    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=0.001)



    for epoch in range(num_epochs):
        metric = Accumulator(4)
        epoch_loss_sum = 0.0
        for i,(features,labels) in enumerate(train_iter):
            batch_size = features.shape[0]
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(features)
            l = loss(outputs, labels).mean(1).mean(1).sum()
            l.backward()
            optimizer.step()
        
            # 切断梯度计算
            with torch.no_grad():
                print(f"batch loss: {l/batch_size:.6f}")
                epoch_loss_sum += l/batch_size
                # train_acc_sum += accuracy(outputs, labels)   
        epoch_loss_sum /= len(train_iter)
        # train_acc_sum /= len(train_iter)
        
        print(f"======================= epoch {epoch + 1} =======================")
        print(f"epoch_loss: {epoch_loss_sum:.6f}")


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
    # 设置超参数
    batch_size = 32
    num_epochs = 10
    num_classes = 21
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # 加载数据集
    crop_size = (320, 480)
    train_iter, test_iter = load_data_voc(batch_size, crop_size)




    # 定义模型
    pretrained_net = models.resnet18(weights="IMAGENET1K_V1")
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, padding=16, stride=32))
    # 初始化上采样卷积层的权重
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)

    # 训练模型
    train(net, train_iter,test_iter,num_epochs,device)

    # 保存模型
    torch.save(net.state_dict(), './pth/resnet18_segmentation.pth')



if __name__ == '__main__':
    main()




