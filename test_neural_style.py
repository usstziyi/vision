import torch
import torchvision
from torch import nn
from matplotlib import pyplot as plt
from PIL import Image







rgb_mean = torch.tensor([0.485, 0.456, 0.406]) # （3,)
rgb_std = torch.tensor([0.229, 0.224, 0.225]) # （3,)

# img plt(3, H, W)
def preprocess(img, resize_shaple):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(resize_shaple),
        torchvision.transforms.ToTensor(), # 转 Tensor 并缩放到 [0, 1]
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    # (1,3,H',W')
    return transforms(img).unsqueeze(0)

# img(1, 3, H, W)
def postprocess(img):
    # img(3, H, W)
    img = img[0].to('cpu')
    # 反归一化后，像素值可能会因为浮点数误差或模型预测超出范围而变成小于 0 或大于 1 的数。
    # clamp 函数将所有值强制限制在 [0, 1] 区间内，确保它们是有效的像素强度值（0代表黑，1代表白）。
    # 第一步：将通道维度移动到最后(H, W, 3)
    # 第二步：将像素值从归一化后的范围转换回原始范围(广播操作)
    # 第三步：将像素值限制在 [0, 1] 区间内
    # img(H, W, 3)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    # ToPILImage() 需要CHW格式,返回HWC
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))


content_layers = [25]
style_layers = [0, 5, 10, 19, 28]


# 提取内容特征
# 预处理[1, 3, 300, 450]
def get_contents(net, content_img, content_layers):
    # 提取内容特征[1, 512, 37, 56]
    contents = []
    feature = content_img
    for i in range(len(net)):
        feature = net[i](feature)
        if i in content_layers:
            contents.append(feature)
    return contents

# 提取风格特征
# style_img[1, 3, 300, 450]
def get_styles(net, style_img, style_layers):
    # 提取风格特征[1, 64, 300, 450]
    styles = []
    feature = style_img
    for i in range(len(net)):
        feature = net[i](feature)
        if i in style_layers:
            styles.append(feature)
    return styles

def get_synthesized(net, synthesized_img, content_layers, style_layers):
    contents = []
    styles = []
    feature = synthesized_img
    for i in range(len(net)):
        feature = net[i](feature)
        if i in content_layers:
            contents.append(feature)
        if i in style_layers:
            styles.append(feature)
    return contents, styles



class SynthesizedImage(nn.Module):
    # content_img_shape(1, 3, H, W)
    def __init__(self, content_img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        # 默认开启 requires_grad
        self.weight = nn.Parameter(torch.rand(*content_img_shape))

    # 前向传播: 直接返回权重
    def forward(self):
        # (1, 3, H, W)
        return self.weight

# 内容损失
def content_loss(synthesized_content, content):
    # PyTorch 中 .detach() 的作用是：从计算图中断开该张量，使其不参与梯度计算。
    # content.detach()我们要让 content 不参与梯度回传，因为它是一个固定的目标值，不需要被优化。
    return torch.square(synthesized_content - content.detach()).mean()  

# 计算格拉姆矩阵
# X(1, C, H, W)
def gram(X):
    # C
    num_channels = X.shape[1]
    # CHW//C=H*W
    n = X.numel() // X.shape[1]
    # X(C, H*W)
    X = X.reshape((num_channels, n))
    # X(C, H*W) * X(H*W, C)=X(C, C)
    return torch.matmul(X, X.T) / (num_channels * n)


# # X(1, C, H, W)
# def gram(X):
#     # X(C,H,W)
#     X = X.squeeze(0)
#     # X(C,H*W)
#     X = X.reshape(X.shape[0],-1)
#     # X(C,C)
#     X = torch.matmul(X,X.T)/X.numel()
#     # X(C,C)
#     return X


# 风格损失
# synthesized_style(1, C, H, W)
# style(C,C)
def style_loss(synthesized_style, style):
    # detach() 从计算图中断开该张量，使其不参与梯度计算
    return torch.square(gram(synthesized_style) - gram(style.detach())).mean()


# 总变分损失
# synthesized_img(1, C, H, W)
# TV Loss=0.5×(Vertical Diff+Horizontal Diff)
# 在训练过程中，最小化这个 Loss 会迫使网络输出的相邻像素值尽可能接近。
# Total Variation: 总变分，直观地说，它计算的是图像中所有相邻像素之间差异的总和。
def tv_loss(synthesized_img):
    return 0.5 * (torch.abs(synthesized_img[:, :, 1:, :] - synthesized_img[:, :, :-1, :]).mean() + torch.abs(synthesized_img[:, :, :, 1:] - synthesized_img[:, :, :, :-1]).mean())



# 封装损失函数
# synthesized_img(1, C, H, W)
def compute_loss(synthesized_img, list_synthesized_contents, list_synthesized_styles, list_contents, list_styles):
    content_weight, style_weight, tv_weight = 1, 1e3, 10
    # 内容损失
    contents_l = [content_loss(synthesized_content, content) * content_weight for synthesized_content, content in zip(list_synthesized_contents, list_contents)]
    # 风格损失
    styles_l = [style_loss(synthesized_style, style) * style_weight for synthesized_style, style in zip(list_synthesized_styles, list_styles)]

    # 全变分损失
    tv_l = tv_loss(synthesized_img) * tv_weight     
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

# content_img(1, 3, H, W)
# contents(1, 512, 37, 56)
# styles(1, 64, 300, 450)
def train(net, content_img, style_img, device, lr, num_epochs, lr_decay_epoch):
    net.to(device)
    # 重要：冻结VGG网络的参数
    net.eval()  # 设置为评估模式
    for param in net.parameters():
        param.requires_grad = False
    
    # 初始化生成图像网络[1, 3, 300, 450]
    synthesized_net = SynthesizedImage(content_img.shape).to(device)
    synthesized_net.weight.data.copy_(content_img.data)
    synthesized_img = synthesized_net()

    # 损失函数
    # 在外部

    # 初始化优化器 - 应该优化合成图像的参数，而不是VGG网络的参数
    optimizer = torch.optim.Adam(synthesized_net.parameters(), lr=lr)  # 修改这里
    # 初始化学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_decay_epoch, 0.8)


    
    # 提取"内容图像"特征
    list_contents = get_contents(net, content_img, content_layers)
    # 提取"风格图像"特征
    list_styles = get_styles(net, style_img, style_layers)

    # 开启交互绘图模式
    plt.ion()
    fig, ax = plt.subplots()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # 提取"生成图像" 图像特征 风格特征
        list_synthesized_contents, list_synthesized_styles = get_synthesized(net, synthesized_img, content_layers, style_layers)
        # 计算损失
        contents_l, styles_l, tv_l, l = compute_loss(synthesized_img, list_synthesized_contents, list_synthesized_styles, list_contents, list_styles)
        l.backward()
        optimizer.step()
        scheduler.step() # 更新学习率调度器，调整下一轮的学习率
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {l.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 显示当前合成图像
        if epoch % 10 == 0:
            img = postprocess(synthesized_img)
            ax.imshow(img)
            ax.axis('off')
            plt.pause(0.01)

    # 关闭交互绘图模式
    plt.ioff()
    plt.tight_layout()
    plt.show()




def main():
    # 定义超参数
    lr = 0.3
    num_epochs = 500
    lr_decay_epoch = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)


    # 加载数据
    content_img = Image.open('./img/rainier.jpg')
    style_img = Image.open('./img/autumn-oak.jpg')

    print(content_img.size, style_img.size)




    # 预处理[1, 3, 300, 450]
    content_img = preprocess(content_img, (300, 450)).to(device)
    style_img = preprocess(style_img, (300, 450)).to(device)



    # 定义模型
    pretrained_net = torchvision.models.vgg19(weights='IMAGENET1K_V1')
    net = nn.Sequential(*[pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)])





    train(net, content_img, style_img, device, lr, num_epochs, lr_decay_epoch)


if __name__ == '__main__':
    main()