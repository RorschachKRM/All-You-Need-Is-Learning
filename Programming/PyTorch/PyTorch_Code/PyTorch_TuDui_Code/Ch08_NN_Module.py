import torch.nn as nn
import torch


"""
nn.Module是所有神经网络模块的基类。
格式：
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()  # 必需初始化
        # ......

    def forward(self, input):
        # ......
        # return ...
"""

class EasyNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

MyNN = EasyNN()
x = torch.tensor(1.)
output = MyNN(x)