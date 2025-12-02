import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from common import TinySSD



def main():
    net = TinySSD(num_classes=1)
    # print(net)
    X = torch.zeros((32, 3, 256, 256))
    anchors, cls_preds, bbox_preds = net(X)

    print("-----")
    print('output anchors:', anchors.shape)
    print('output class preds:', cls_preds.shape)
    print('output bbox preds:', bbox_preds.shape)

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