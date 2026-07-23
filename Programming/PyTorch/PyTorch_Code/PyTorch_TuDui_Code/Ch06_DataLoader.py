import torchvision

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor())

"""
数据加载器结合了数据集和采样器，并提供一个可迭代对象，用于访问给定的数据集。
常用参数参数：
    dataset（Dataset）：要从中加载数据的数据集。

    batch_size（int，可选）：每个批次加载的样本数（默认值：1）。
    
    shuffle（bool，可选）：设置为True以在每个 epoch 重新打乱数据顺序（默认值：False）。

    num_workers (int，可选)：用于数据加载的子进程数量。0 表示数据将在主进程中加载。（默认值：0）
    
    drop_last (bool，可选)：如果数据集大小不能被分批大小整除，则设置为 True 以丢弃最后一个不完整的批次。如果为 False 且数据集大小不能被分批大小整除，则最后一个批次将较小。（默认值：False）
"""
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter("dataloader")
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step = step + 1

writer.close()