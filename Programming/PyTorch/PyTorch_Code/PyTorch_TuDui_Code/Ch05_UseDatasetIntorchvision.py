import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=dataset_transform, download=True)


# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.shape

writer = SummaryWriter("logs_CIFAR10")
for i in range(10):
    img,target = train_set[i]
    writer.add_image("train_set", img, i)

writer.close()