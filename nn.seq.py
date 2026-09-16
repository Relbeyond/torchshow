import torch
from torch import nn
from torch.nn import Conv2d, Flatten, Linear, Sequential
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

from P10 import writer


class Tudui(nn.Module):
    def __init__(self): # 初始化父类属性
        super(Tudui, self).__init__() # 继承父类初始化方法
        # self.conv1 = Conv2d(3, 32, 5, padding=2) # 卷积核大小为5*5
        # self.maxpool1 = MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32 ,5, padding=2) # 卷积核大小为5*5
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.mode1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),


            )

    def forward(self, x):
        x = self.mode1(x)
        return x

tudui = Tudui()
print(tudui)
input = torch.ones((64, 3, 32, 32))
output = tudui(input)
print(output.shape)

writer = SummaryWriter("logs_seq")
writer.add_graph(tudui, input)
writer.close()
