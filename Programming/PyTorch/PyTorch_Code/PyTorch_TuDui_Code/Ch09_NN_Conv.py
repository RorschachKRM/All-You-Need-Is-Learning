import torch.nn as nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

"""
Conv2d在由多个输入平面组成的输入信号上应用二维卷积。

class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

参数：
    in_channels (int) – 输入图像中的通道数
    out_channels (int) – 卷积产生的通道数
    kernel_size (int 或 tuple) – 卷积核的大小
    stride (int 或 tuple, 可选) – 卷积的步幅，默认为：1
    padding (int, tuple 或 str, 可选) – 填充，默认值：0
        =valid=padding=0
        =same=padding=(K - 1) / 2  (stride=1，且K为奇数时)
    dilation (int 或 tuple, 可选) – 空洞率,空洞卷积算法,核元素之间的间距，默认为：1
    groups (int, 可选) – 从输入通道到输出通道的阻塞连接数，默认为：1
    bias (bool, 可选) – 如果为 True，则向输出添加可学习的偏置，默认为：True
    padding_mode (str, 可选) – 'zeros'、'reflect'、'replicate' 或 'circular'，默认为：'zeros'

形状：
    输入：(N, Cin​, Hin​, Win​) 或 (Cin​, Hin​, Win​)
    输出：(N, Cout​, Hout​, Wout​) 或 (Cout​, Hout​, Wout​)
        完整公式：Output = ⌊ (Input + 2*P - D*(K - 1) - 1) / S + 1 ⌋
        常用：Output = (Input + 2P - K) / S + 1
                = Input + 2P - K + 1 （步长=1时）
"""

test_dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=False)

test_dataloader = DataLoader(test_dataset, batch_size=64)

class EasyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        output = self.conv1(x)
        return output

MyEasyCNN = EasyCNN()

writer = SummaryWriter("logs_CIFAR10")

step = 0

"""
enumerate() 用来在遍历数据时，同时拿到序号和当前元素
test_dataloader返回的元素：(imgs, labels)
enumerate(test_dataloader)返回的元素：(0, (imgs_第1批, labels_第1批))
step则拿到了当前是第几个batch
"""
for step, (imgs, labels) in enumerate(test_dataloader):
    # imgs.shape
    writer.add_images("imgs", imgs, step)
    conv_imgs = MyEasyCNN(imgs)
    # conv_imgs.shape

    vi_conv_imgs = torch.reshape(conv_imgs, (-1, 3, 30, 30 ))
    writer.add_images("vi_conv_imgs", vi_conv_imgs, step)

writer.close()

