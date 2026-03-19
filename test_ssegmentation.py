import os
import torch
import torchvision

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
    # shape(H,W)=(281,500)
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
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx], *self.crop_size)
        # 返回特征和标签，标签的形状为(281,500)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)


def load_data_voc(batch_size, crop_size):
    """加载VOC语义分割数据集"""
    voc_dir = "./data/VOCdevkit/VOC2012"
    num_workers = 0
    train_iter = torch.utils.data.DataLoader(VOCSegDataset(True, crop_size, voc_dir), batch_size,shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(VOCSegDataset(False, crop_size, voc_dir), batch_size,drop_last=True, num_workers=num_workers)
    return train_iter, test_iter


def main():
    batch_size = 64
    crop_size = (320, 480)
    train_iter, test_iter = load_data_voc(batch_size, crop_size)
    print(train_iter)
    print(test_iter)


if __name__ == '__main__':
    main()