---
title: 上采样
tags:
  - DDPM
  - Convolution
  - UpSample
related:
  - "[[CNN知识点]]"
---
# 1. 最近邻插值+卷积实现
此处是使用最近邻插值x2 + 3×3 卷积实现上采样：
（代码来源：\DDPM\ddpm_model.py）
```python
class UpSample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x
```
## 关键讲解`F.interpolate`:
属于 `torch.nn.functional`（常用别名 `F`），是 PyTorch 中的**张量插值/缩放函数**，专门用于改变输入张量的空间尺寸（高和宽），可以放大（上采样）或缩小（下采样）。

```python
x = F.interpolate(x, scale_factor=2, mode="nearest")
```
x：输入张量，形状为 `(B, C, H, W)` — 批次、通道、高、宽
scale_factor=2：将空间维度（H 和 W）**各放大 2 倍**。`H→2H`，`W→2W`
mode="nearest"：使用**最近邻插值**算法来填充新增的像素
	对比 `mode="bilinear"`：双线性插值会取周围 2×2 像素的加权平均，输出更平滑但计算更贵

为什么 DDPM / U-Net 上采样故意不用 Bilinear 而用 Nearest：
1. 避免引入 checkerboard artifacts（棋盘伪影）
    - Bilinear 插值在上采样时会产生类似卷积转置（transposed convolution）的棋盘格重叠问题
    - Nearest 是"刚性"复制，不会产生灰度渐变的过渡带，配合后续卷积反而更容易学习出干净的纹理
2. 后续的 `3×3 Conv` 负责"修复"粗糙度
	- 整个上采样模块的设计是 **Nearest 上采样 → Conv2d**：
	```python
	x = F.interpolate(x, scale_factor=2, mode="nearest")  # 粗糙放大
	x = self.conv(x)  # 3×3 卷积平滑+特征提取
	```
	- Nearest 只是尺寸放大，不引入额外噪声
	- 卷积层有**可学习的参数**，它会自己学出最优的"平滑+特征重组"策略
	- 这种组合在生成模型中比 `mode="bilinear"` 效果更好，训练也更稳定
3. **计算效率**：Nearest 几乎零计算量（只是索引查找），比 Bilinear 快。