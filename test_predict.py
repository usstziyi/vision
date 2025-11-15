from common import multibox_detection
from common import show_anchors
import torch
from d2l import torch as d2l
torch.set_printoptions(2) 



def main():
    img = d2l.plt.imread('./img/catdog.jpg')
    h, w = img.shape[:2]
    anchor_scale = torch.tensor((w, h, w, h))

    anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92],
                            [0.08, 0.2, 0.56, 0.95],
                            [0.15, 0.3, 0.62, 0.91], 
                            [0.55, 0.2, 0.9, 0.88]])

    # 第一个画布：显示所有锚框
    fig1 = d2l.plt.figure(1)
    d2l.plt.imshow(img)
    show_anchors(d2l.plt.gca(), anchors * anchor_scale, ['0:dog=0.9', '1:dog=0.8', '2:dog=0.7', '3:cat=0.9'])

    # offset_preds(NAC*4)
    offset_preds = torch.tensor([0] * anchors.numel()) # numel() 是PyTorch张量的一个方法，返回张量中元素的总数

    # cls_probs(NCLS, NAC)
    cls_probs = torch.tensor([[0,   0,   0,   0],  # 背景的预测概率
                            [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                            [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率


    # 非极大值抑制
    # (class_id, scores, predicted_bb)
    output = multibox_detection(cls_probs.unsqueeze(dim=0),
                                offset_preds.unsqueeze(dim=0),
                                anchors.unsqueeze(dim=0),
                                nms_threshold=0.5)
    print(output)

    # 第二个画布：显示NMS后的结果
    fig2 = d2l.plt.figure(2)
    d2l.plt.imshow(img)
    for i in output[0].detach().numpy():
        if i[0] == -1:
            continue
        label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
        show_anchors(d2l.plt.gca(), [torch.tensor(i[2:]) * anchor_scale], label)
    d2l.plt.show()


if __name__ == '__main__':
    main()
