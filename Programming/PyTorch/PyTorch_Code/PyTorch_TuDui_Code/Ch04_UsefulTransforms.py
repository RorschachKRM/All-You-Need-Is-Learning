from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("IMG/IKUN.JPG")

# ToTensor
"""
ToTensor干两件事：
    1. 把 PIL Image / numpy 数组转为 torch.Tensor
    2. PIL Image 时：像素值从 [0, 255] 整数 → [0.0, 1.0] 浮点数（除以 255）。（对图片默认标准化）
    numpy 数组时：保持原值不变（不除以 255）
"""
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("totensor", img_tensor)

# Nomalize
"""
Nomalize是使用均值和标准差对张量图像进行逐通道归一化。此变换不支持 PIL 图像。注意:此变换的作用方式与输入张量不同，即它不会改变输入张量本身。
给定均值：(mean[1],...,mean[n])和标准差：(std[1],..,std[n])，对于n个通道。
此变换将对输入张量的每个通道进行归一化：
output[channel] = (input[channel] - mean[channel]) / std[channel]

参数：
    mean（序列）：每个通道的均值序列。  
    std（序列）：每个通道的标准差序列。
    inplace（布尔值，可选）：是否执行原地操作。
"""
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([2.5, 2.5, 2.5], [2.5, 2.5, 2.5]) 
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("normtensor", img_norm, 3)


# Resize
"""
将输入图像（PIL、tensor类型都行）调整为指定大小。
如果图像是 Torch Tensor，则预期其形状为 [..., H, W]，其中 ... 表示最多两个前导维度。

参数：
    size（序列或整数）：期望的输出大小。如果 size 是一个序列，例如(h, w)，则输出大小将与此匹配。如果 size 是一个整数，则图像的较小边缘将与此数字匹配，要保持原图长宽比。例如，如果高度 > 宽度，则图像将被缩放为(size * height / width, size)。
"""
trans_resize = transforms.Resize([90, 184])
img_resize = trans_resize(img_tensor)
writer.add_image("resize", img_resize, 2)

# Compose
"""
将多个变换组合在一起。示例：
transforms.Compose([transforms.CenterCrop(10), transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float),])

注意:要编写变换脚本，请使用torch.nn.Sequential
"""


# Compose - Resize - 2
trans_resize_2 = transforms.Resize(66)
transforms_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = transforms_compose(img)
writer.add_image("compose_resize", img_resize_2, 0)



writer.close()