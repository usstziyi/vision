import torchvision.models as models
import torch 



def display_net(net):

    for name, child in net.named_children():
        print(f"子模块的名字: {name}, 类型: {type(child).__name__}")


def main():
    net = models.resnet18()
    display_net(net)


if __name__ == "__main__":
    main()