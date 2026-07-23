
import os
    # os.path.join()
    # os.listdir()
import numpy as np
    # np.array()
from PIL import Image
    # Image.open() - 返回:PIL.Image 对象，RGB，ToTensor() 行为值除以 255 → [0.0, 1.0]
import cv2
    # cv2.imread() - 返回:numpy.ndarray数组，BGR，ToTensor() 行为原值不变（保持 [0, 255]）


import torch
    # torch.tensor()


import torchvision
    # torchvision.datasets.CIFAR10()
from torchvision import transforms
    # transforms.ToTensor()
    # transforms.Normalize()
    # transforms.Compose()
    # transforms.Resize()


from torch.utils.data import Dataset
    # class MyDataset(Dataset)
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter("logs")


import torch.nn as nn
    # nn.Module
    # nn.Conv2d()
import torch.nn.functional as F
    # F.relu()