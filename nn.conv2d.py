from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P10 import writer

data_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)


dataloader = DataLoader(data_set, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__() # 继承父类
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("logs")

step = 0

for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    print(imgs.shape)
    print(output.shape)
    #
    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 1, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1


