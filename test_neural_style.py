import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt

# X(1, 3, H, W)
# contents_Y(1, 512, 37, 56)
# styles_Y(1, 64, 300, 450)
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    # X(1, 3, H, W)
    X, styles_Y_gram, optimizer = get_inits(X, device, lr, styles_Y)
    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8)
    # 初始化动画
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))
    # 训练模型
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        contents_Y_hat, styles_Y_hat = extract_features(X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        l.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)), float(sum(styles_l)), float(tv_l)])
    return X