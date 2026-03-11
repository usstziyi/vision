import os
import torch
from common import show_boxes
from common import load_data_bananas


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

