import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P10 import writer

test_data = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform= torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")

for epoch in range(2):
    step = 0
    for data in test_loader: #data是一个batch的数据,data会接收test_loader中的数据
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        # `{}` 是占位符，`.format(epoch)` 把变量 `epoch` 的值填到大括号位置。
        writer.add_images("Epoch:{}".format(epoch), imgs, step)
        step = step + 1

writer.close()