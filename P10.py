import torchvision
from torch.utils.tensorboard import SummaryWriter

# torchvision.transforms.ToTensor()将图片转换为张量
dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
# torchvision.datasets.CIFAR10()下载CIFAR10数据集
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=dataset_transform, download=True)

#print(test_set[0])

writer = SummaryWriter("P10")
for i in range(10):
    img, label = test_set[i]
    writer.add_image("test_set", img, i)

writer.close()