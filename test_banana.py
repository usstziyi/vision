import os
import pandas as pd
import torch
import torchvision.io as tv_io
from torch.utils.data import Dataset, DataLoader
from common import show_boxes

def read_labels_bananas(is_train=True):
    """仅读取标签CSV，不碰图片"""
    data_dir = './data/banana-detection'
    split_dir = 'bananas_train' if is_train else 'bananas_val'
    csv_fname = os.path.join(data_dir, split_dir, 'label.csv')
    csv_data = pd.read_csv(csv_fname) # DataFrame
    csv_data = csv_data.set_index('img_name')
    #           label  xmin  ymin  xmax  ymax
    # img_name
    # 0.png         0   104    20   143    58
    # 1.png         0    68   175   118   223
    # 2.png         0   163   173   218   239
    # 3.png         0    48   157    84   201
    # 4.png         0    32    34    90    86
        
    # 返回标签数据和图像目录路径
    img_dir = os.path.join(data_dir, split_dir, 'images')
    return csv_data, img_dir


class BananasDataset(Dataset):
    """
    真正的懒加载版本：
    1. __init__ 完全不读取任何图片文件。
    2. __getitem__ 只读取一次图片，同时完成尺寸获取和归一化。
    """
    def __init__(self, is_train, normalize_coords=True):
        self.labels_df, self.img_dir = read_labels_bananas(is_train)
        self.img_names = list(self.labels_df.index)
        self.normalize_coords = normalize_coords

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, f'{img_name}')
        
        # 【关键】只在这里读取一次图片
        image = tv_io.read_image(img_path).float()  # (C, H, W)
        
        # 直接从读取到的 image tensor 获取宽高，无需预存！
        # 可以应对不同尺寸的图片
        # 现代解法：现在的检测网络（如 YOLO, SSD, Faster R-CNN）通常使用
        # 全局平均池化 (Global Average Pooling) 或 ROI Align 来替代全连接层，从而彻底解决对固定尺寸的依赖。
        _, h, w = image.shape 
        
        # 获取标签 [class, xmin, ymin, xmax, ymax]
        # .copy() 确保我们操作的是数据的副本，避免 pandas 的 SettingWithCopyWarning
        target = self.labels_df.loc[img_name].values.astype('float32').copy()
        
        # 可选：坐标归一化
        if self.normalize_coords:
            # 使用当前这张图的真实宽高进行归一化
            target[1] /= w   # xmin
            target[2] /= h   # ymin
            target[3] /= w   # xmax
            target[4] /= h   # ymax
        
        # 转换为 tensor 并增加维度 -> (1, 5)
        target_tensor = torch.tensor(target).unsqueeze(0)
        # image: (C, H, W)
        # target_tensor: (1, 5)
        return image, target_tensor

    def __len__(self):
        return len(self.img_names)

def load_data_bananas(batch_size, normalize_coords=True):
    train_dataset = BananasDataset(is_train=True, normalize_coords=normalize_coords)
    val_dataset = BananasDataset(is_train=False, normalize_coords=normalize_coords)
    
    # num_workers > 0 可以进一步加速数据加载（多进程）
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_iter = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"训练集批次数量：{len(train_iter)}")    # batch_number
    print(f"训练集样本数量：{len(train_dataset)}") # sample_number
    print(f"验证集批次数量：{len(val_iter)}")
    print(f"验证集样本数量：{len(val_dataset)}")
    return train_iter, val_iter

if __name__ == '__main__':
    train_iter, val_iter = load_data_bananas(32)
    batch = next(iter(train_iter))
    # batch[0]: (batch_size, C, H, W)
    # batch[1]: (batch_size, 1, 5)
    print(batch[0].shape, batch[1].shape)

    # 用matplotlib可视化前10张图片
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i in range(10):
        ax = axes[i // 5, i % 5]
        img = batch[0][i].permute(1, 2, 0).numpy() / 255.0  # (H, W, C)
        h, w, _ = img.shape
        scale = torch.tensor([w, h, w, h])

        gt_class = batch[1][i][0][0] # tensor
        gt_box = batch[1][i][0][1:] # tensor
        # 绘制真实框
        show_boxes(ax, gt_box * scale.unsqueeze(0), linewidth=1, colors=['red'])
        ax.set_title(f'class:{int(gt_class.item())}')
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

