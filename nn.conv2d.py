from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P10 import writer

# 创建CIFAR10数据集，不进行训练，转换为Tensor格式，并下载到本地
data_set = torchvision.datasets.CIFAR10(
    root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)


# 创建数据加载器，批量大小为64
dataloader = DataLoader(data_set, batch_size=64)

# 定义一个自定义神经网络类Tudui，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__() # 继承父类初始化方法
        # 定义一个二维卷积层，输入3通道，输出6通道，卷积核大小为3，步长为1，填充为0
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)

    # 定义前向传播方法
    def forward(self, x):
        # 将输入数据通过卷积层处理
        x = self.conv1(x)
        return x

# 创建Tudui类的实例
tudui = Tudui()

# 创建TensorBoard写入器，日志保存在logs目录
writer = SummaryWriter("logs")

# 初始化步骤计数器
step = 0

# 遍历数据加载器
for data in dataloader:
    # 获取批次数据和标签
    imgs, targets = data
    # 将数据通过神经网络模型处理
    output = tudui(imgs)
    # 打印输入和输出的形状
    print(imgs.shape)
    print(output.shape)
    # 将输入图像写入TensorBoard
    writer.add_images("input", imgs, step)

    # 调整输出形状以匹配图像格式，并写入TensorBoard
    output = torch.reshape(output, (-1, 1, 30, 30))
    writer.add_images("output", output, step)
    # 增加步骤计数
    step = step + 1


