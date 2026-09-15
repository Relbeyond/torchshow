import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P10 import writer

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)


dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):     # 继承nn.Module
    def __init__(self):     # 初始化父类属性
        super().__init__()  # 调用父类初始化方法
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, ceil_mode=False) # ceil_mode=True表示向上取整


    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()

writer = SummaryWriter("logs_axpool")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()

