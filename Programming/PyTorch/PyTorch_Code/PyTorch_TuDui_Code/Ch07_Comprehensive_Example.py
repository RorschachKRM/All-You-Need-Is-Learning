"""
=============================================================================
Ch01~06 综合实战示例：完整的图像分类数据流水线
=============================================================================
整合知识点：
  Ch01 - 自定义 Dataset 类（继承 torch.utils.data.Dataset）
  Ch02 - TensorBoard 可视化（SummaryWriter：图像 + 标量）
  Ch03 - transforms.ToTensor() 图像→张量转换
  Ch04 - transforms 工具链：Normalize / Resize / Compose
  Ch05 - torchvision 内置数据集（CIFAR10）
  Ch06 - DataLoader 批量加载（batch_size / shuffle / num_workers / drop_last）

场景说明：
  我们同时处理两个数据集 ——
  1. 自定义的"蚂蚁 vs 蜜蜂"二分类数据（Ch01 的 ants/bees）
  2. torchvision 内置的 CIFAR-10 十分类数据（Ch05）

  最终用 DataLoader（Ch06）统一加载，并将训练过程中的图像、损失、
  准确率等全部记录到 TensorBoard（Ch02）中可视化。
=============================================================================
"""

import os
import numpy as np
from PIL import Image

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# ============================================================================
# 第1部分：自定义 Dataset（Ch01）- 蚂蚁 vs 蜜蜂
# ============================================================================
class AntsBeesDataset(Dataset):
    """
    自定义数据集：读取 ants_image / bees_image 两个文件夹下的图片。

    Ch01 知识点回顾：
      - 必须继承 torch.utils.data.Dataset
      - 必须实现 __init__、__getitem__、__len__
      - __getitem__ 返回 (img, label)，其中 label 可用整数索引
      - 支持 + 运算符拼接两个数据集
    """

    def __init__(self, root_dir, label_name, label_idx, transform=None):
        """
        root_dir   : 数据集根目录（如 "Data/train"）
        label_name : 类别文件夹名（如 "ants_image"）
        label_idx  : 类别对应的整数标签（如 0 代表 ants）
        transform  : torchvision transforms 组合（Ch04 知识点）
        """
        self.root_dir = root_dir
        self.label_name = label_name
        self.label_idx = label_idx
        self.transform = transform
        self.img_dir = os.path.join(root_dir, label_name)
        self.img_list = os.listdir(self.img_dir)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")  # 统一转为 RGB 三通道
        label = self.label_idx

        if self.transform:
            img = self.transform(img)  # Ch04：应用 transforms

        return img, label

    def __len__(self):
        return len(self.img_list)


# ============================================================================
# 第2部分：定义 transforms 流水线（Ch03 + Ch04）
# ============================================================================
"""
Ch03 & Ch04 知识点回顾：
  - transforms.ToTensor()：PIL Image [0,255] → Tensor [0.0, 1.0]，同时 HWC→CHW
  - transforms.Normalize(mean, std)：逐通道标准化 output = (input - mean) / std
  - transforms.Resize(size)：缩放图像；传入 (H,W) 为固定尺寸，传入 int 为短边等比缩放
  - transforms.Compose([...])：将多个变换串联成一个流水线
  - transforms.RandomHorizontalFlip()：数据增强——随机水平翻转
"""

# 自定义数据集的 transforms：缩放到统一尺寸 → 转为张量 → 标准化
custom_transform = transforms.Compose([
    transforms.Resize((128, 128)),                # Ch04：固定缩放到 128×128
    transforms.RandomHorizontalFlip(p=0.5),       # 数据增强：50% 概率水平翻转
    transforms.ToTensor(),                        # Ch03：PIL → Tensor [0,1]
    transforms.Normalize(                         # Ch04：三通道标准化
        mean=[0.485, 0.456, 0.406],              # ImageNet 均值（迁移学习常用）
        std=[0.229, 0.224, 0.225]                # ImageNet 标准差
    ),
])

# CIFAR-10 的 transforms（不需要 Resize，CIFAR-10 本身就是 32×32）
cifar10_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],           # CIFAR-10 专用均值/标准差
        std=[0.2023, 0.1994, 0.2010]
    ),
])

# 仅 ToTensor 版本（用于 TensorBoard 可视化，不需要标准化）
visual_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# ============================================================================
# 第3部分：构建数据集
# ============================================================================

# ---- 3.1 自定义数据集（Ch01）----
print("=" * 60)
print("构建 自定义数据集（Ch01）+ 内置数据集（Ch05）")
print("=" * 60)

root_dir = "Data/train"

ants_dataset = AntsBeesDataset(root_dir, "ants_image", label_idx=0, transform=custom_transform)
bees_dataset = AntsBeesDataset(root_dir, "bees_image", label_idx=1, transform=custom_transform)

# Ch01：用 + 拼接两个 Dataset
custom_train_set = ants_dataset + bees_dataset
print(f"✅ 自定义数据集加载完成：{len(ants_dataset)} 张蚂蚁 + {len(bees_dataset)} 张蜜蜂")
print(f"   拼接后共 {len(custom_train_set)} 张图片")

# 同样构建一个不标准化的版本用于 TensorBoard 可视化
ants_viz = AntsBeesDataset(root_dir, "ants_image", label_idx=0, transform=visual_transform)
bees_viz = AntsBeesDataset(root_dir, "bees_image", label_idx=1, transform=visual_transform)
custom_viz_set = ants_viz + bees_viz

# ---- 3.2 torchvision 内置数据集（Ch05）----
"""
Ch05 知识点回顾：
  - torchvision.datasets.CIFAR10(root, train, transform, download)
  - train=True  → 训练集（50000 张）
  - train=False → 测试集（10000 张）
  - dataset.classes → 类别名称列表
  - dataset[i]     → 返回 (img_tensor, target_int)
"""
print(f"\n📥 正在加载 CIFAR-10 数据集...")
train_set = torchvision.datasets.CIFAR10(
    root="./CIFAR10",
    train=True,
    transform=cifar10_transform,
    download=True,
)
test_set = torchvision.datasets.CIFAR10(
    root="./CIFAR10",
    train=False,
    transform=cifar10_transform,
    download=True,
)

print(f"✅ CIFAR-10 加载完成：训练集 {len(train_set)} 张，测试集 {len(test_set)} 张")
print(f"   类别列表：{train_set.classes}")

# 用于可视化的 CIFAR-10（不标准化）
cifar10_viz_set = torchvision.datasets.CIFAR10(
    root="./CIFAR10",
    train=False,
    transform=transforms.ToTensor(),  # 只转张量不标准化
)


# ============================================================================
# 第4部分：DataLoader 批量加载（Ch06）
# ============================================================================
"""
Ch06 知识点回顾：
  - DataLoader(dataset, batch_size, shuffle, num_workers, drop_last)
  - batch_size  : 每批加载多少样本
  - shuffle     : 每个 epoch 是否打乱数据
  - num_workers : 子进程数量（0 = 主进程加载）
  - drop_last   : 是否丢弃最后不足 batch_size 的批次
  - 遍历 DataLoader 返回 (imgs, targets)，形状为 [B, C, H, W] 和 [B]
"""

print(f"\n📦 构建 DataLoader...")

# 自定义数据集的 DataLoader
custom_loader = DataLoader(
    dataset=custom_train_set,
    batch_size=16,
    shuffle=True,       # 每 epoch 打乱
    num_workers=0,      # Windows 下建议 0
    drop_last=True,     # 丢弃不足 16 的最后一批
)

# CIFAR-10 训练集的 DataLoader
train_loader = DataLoader(
    dataset=train_set,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=False,    # 保留最后不完整的批次
)

# CIFAR-10 测试集的 DataLoader
test_loader = DataLoader(
    dataset=test_set,
    batch_size=64,
    shuffle=False,      # 测试集不需要打乱！
    num_workers=0,
    drop_last=False,
)

print(f"✅ DataLoader 构建完成：")
print(f"   自定义 Loader：{len(custom_loader)} 批次 (batch_size=16, drop_last=True)")
print(f"   CIFAR-10 训练 Loader：{len(train_loader)} 批次 (batch_size=64)")
print(f"   CIFAR-10 测试 Loader：{len(test_loader)} 批次 (batch_size=64)")


# ============================================================================
# 第5部分：TensorBoard 可视化（Ch02）
# ============================================================================
"""
Ch02 知识点回顾：
  - SummaryWriter("logs_dir")   : 创建日志写入器
  - writer.add_image(tag, img, step, dataformats) : 写入单张图像
  - writer.add_images(tag, imgs, step)            : 写入批量图像（NCHW）
  - writer.add_scalar(tag, value, step)            : 写入标量（损失、准确率等）
  - writer.close()                                 : 关闭写入器
"""
print(f"\n📊 初始化 TensorBoard SummaryWriter...")
writer = SummaryWriter("logs_comprehensive")

# ---- 5.1 可视化原始图像（Ch02）----
# 用不标准化的数据展示原图，标准化后的图人眼无法辨认
print("   记录自定义数据集样本图像到 TensorBoard...")
viz_loader = DataLoader(custom_viz_set, batch_size=8, shuffle=True)

# add_images 要求 [N, C, H, W] 格式（Ch02）
data_iter = iter(viz_loader)  
sample_imgs, sample_labels = next(data_iter)  # 只想取一个 batch
writer.add_images("Custom_AntsBees/Samples", sample_imgs, 0)

print("   记录 CIFAR-10 样本图像到 TensorBoard...")
for i in range(10):
    img, target = cifar10_viz_set[i]
    writer.add_image(
        f"CIFAR10/{cifar10_viz_set.classes[target]}",
        img, 0
    )

# ---- 5.2 模拟训练过程并记录标量（Ch02：add_scalar）----
"""
这里模拟一个训练过程，展示如何用 TensorBoard 记录 loss 和 accuracy。
实际训练时，这些值来自模型的 forward + loss 计算。
"""

print("\n🔄 模拟 CIFAR-10 训练过程，并记录到 TensorBoard...")
num_epochs = 3

for epoch in range(num_epochs):
    print(f"   Epoch {epoch+1}/{num_epochs}...")

    # ---- 训练阶段 ----
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (imgs, targets) in enumerate(train_loader):
        # ======== 实际训练时，这里放模型代码 ========
        # outputs = model(imgs)
        # loss = criterion(outputs, targets)
        # loss.backward()
        # optimizer.step()
        # =========================================

        # 模拟：用随机 loss 和准确率演示 TensorBoard 记录
        simulated_loss = np.random.uniform(0.05, 0.8)
        simulated_correct = int(imgs.shape[0] * np.random.uniform(0.3, 0.95))

        running_loss += simulated_loss * imgs.shape[0]
        running_correct += simulated_correct
        total_samples += imgs.shape[0]

        # Ch02：每个批次记录一次 loss（scalar）
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar("CIFAR10_Train/Loss_per_batch", simulated_loss, global_step)
        writer.add_scalar(
            "CIFAR10_Train/Accuracy_per_batch",
            simulated_correct / imgs.shape[0],
            global_step,
        )

    # Ch02：每个 epoch 记录一次平均 loss 和准确率
    epoch_loss = running_loss / total_samples
    epoch_acc = running_correct / total_samples
    writer.add_scalar("CIFAR10_Train/Loss_per_epoch", epoch_loss, epoch)
    writer.add_scalar("CIFAR10_Train/Accuracy_per_epoch", epoch_acc, epoch)
    print(f"      训练 loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")

    # ---- 验证阶段 ----
    total_correct = 0
    total_samples = 0

    for imgs, targets in test_loader:
        # 模拟预测正确数量
        simulated_correct = int(imgs.shape[0] * np.random.uniform(0.35, 0.92))
        total_correct += simulated_correct
        total_samples += imgs.shape[0]

    val_acc = total_correct / total_samples
    writer.add_scalar("CIFAR10_Val/Accuracy", val_acc, epoch)
    print(f"      验证 acc={val_acc:.4f}")

# ---- 5.3 将每个 epoch 的批量图像也写入（Ch06 的 add_images）----
print("\n📷 将 CIFAR-10 测试集的前几个 batch 写入 TensorBoard...")
test_viz_loader = DataLoader(cifar10_viz_set, batch_size=64, shuffle=True)
data_iter = iter(test_viz_loader)
for batch_idx in range(3):  # 只记录前 3 个 batch
    try:
        imgs, targets = next(data_iter)
    except StopIteration:
        break
    writer.add_images(f"CIFAR10_Test_Batches/batch_{batch_idx}", imgs, 0)

writer.close()
print("\n✅ TensorBoard 记录完成！")


# ============================================================================
# 第6部分：汇总输出 —— 展示 DataLoader 的迭代模式（Ch06）
# ============================================================================
print("\n" + "=" * 60)
print("DataLoader 迭代模式演示（Ch06）")
print("=" * 60)

# ---- 展示 DataLoader 的一次迭代 ----
demo_loader = DataLoader(custom_train_set, batch_size=4, shuffle=True, drop_last=True)
imgs, labels = next(iter(demo_loader))

print(f"一个 batch 的图像张量形状：{imgs.shape}")
# 预期：torch.Size([4, 3, 128, 128])
# 解读：batch_size=4, 通道=3(RGB), 高=128, 宽=128（由 Resize 决定）

print(f"一个 batch 的标签形状：{labels.shape}")
# 预期：torch.Size([4])

print(f"标签值：{labels.tolist()}")
# 预期：0 是蚂蚁，1 是蜜蜂

print(f"图像数值范围：[{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
# 预期：标准化后的值，不是 [0,1] 了

# ---- 展示 CIFAR-10 DataLoader ----
cifar_imgs, cifar_labels = next(iter(train_loader))
print(f"\nCIFAR-10 一个 batch 形状：{cifar_imgs.shape}")
# 预期：torch.Size([64, 3, 32, 32])

print(f"CIFAR-10 标签：{cifar_labels[:10].tolist()}...")
print(f"对应类别：{[train_set.classes[l] for l in cifar_labels[:10].tolist()]}...")


# ============================================================================
# 第7部分：关键概念速查表
# ============================================================================
print("\n" + "=" * 60)
print("📋 关键概念速查表")
print("=" * 60)

cheat_sheet = """
┌──────────┬──────────────────────────────────────────────────┐
│  章节    │  核心知识点                                       │
├──────────┼──────────────────────────────────────────────────┤
│  Ch01    │  class MyData(Dataset): __init__, _getitem__,   │
│          │  __len__；返回 (img, label) 元组；支持 + 拼接    │
├──────────┼──────────────────────────────────────────────────┤
│  Ch02    │  SummaryWriter("dir") → add_image(单张)          │
│          │  add_images(批量,NCHW) → add_scalar → close()    │
├──────────┼──────────────────────────────────────────────────┤
│  Ch03    │  transforms.ToTensor(): PIL→Tensor [0,1]         │
│          │  HWC(高宽通道) → CHW(通道高宽)，shape 变化       │
├──────────┼──────────────────────────────────────────────────┤
│  Ch04    │  Normalize(mean,std): 逐通道 (x-μ)/σ              │
│          │  Resize(size): int=短边等比, tuple=固定尺寸      │
│          │  Compose([T1,T2,...]): 串联多个变换              │
├──────────┼──────────────────────────────────────────────────┤
│  Ch05    │  torchvision.datasets.CIFAR10(root,train,        │
│          │  transform,download); dataset.classes 类别名     │
├──────────┼──────────────────────────────────────────────────┤
│  Ch06    │  DataLoader(dataset,batch_size,shuffle,          │
│          │  num_workers,drop_last) → for循环迭代获取batch   │
│          │  返回 (imgs[B,C,H,W], labels[B])                 │
└──────────┴──────────────────────────────────────────────────┘

启动 TensorBoard 查看结果：
    tensorboard --logdir=logs_comprehensive
"""

print(cheat_sheet)
print("🎉 综合示例运行完毕！所有 Ch01~06 的知识点已整合演示。")
