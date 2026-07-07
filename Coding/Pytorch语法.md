---
tags:
  - PyTorch
  - Deep-Learning
related:
  - "[[深度学习基础知识点]]"
---

## 1. 标准导入

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

- `torch`：核心库（张量、数学运算）
- `torch.nn`：网络层、损失函数
- `torch.nn.functional`：无状态的函数式操作（激活函数、填充等）

---

## 2. nn.Module —— 所有网络的基类

```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()       # 必须调用父类初始化
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)
```

- 必须继承 `nn.Module`
- `__init__` 中定义层，`forward` 中定义前向逻辑
- `super().__init__()` 不可省略

---

## 3. nn.Linear —— 线性层 / 全连接层

```python
layer = nn.Linear(in_features=128, out_features=512)
y = layer(x)    # x: (B, 128) → y: (B, 512)
```

数学本质：$y = Wx + b$，内部自动管理权重和偏置。

---

## 4. 常用激活函数

| 函数 | 调用方式 | 特点 |
|------|---------|------|
| ReLU | `F.relu(x)` / `nn.ReLU()` | 最常用，负值截 0 |
| SiLU | `F.silu(x)` / `nn.SiLU()` | 比 ReLU 更平滑 |
| GELU | `F.gelu(x)` / `nn.GELU()` | Transformer 常用 |

`nn.Xxx()` 用于 `nn.Sequential` 中，`F.xxx()` 用于 `forward` 中直接调用。

---

## 5. nn.Sequential —— 串联多个模块

按顺序依次执行每个子模块，前一个的输出自动成为后一个的输入。

```python
self.mlp = nn.Sequential(
    nn.Linear(256, 512),
    nn.SiLU(),
    nn.Linear(512, 256),
)
y = self.mlp(x)   # x → Linear → SiLU → Linear → y
```

**可以混合自定义模块和内置层：**

```python
self.time_mlp = nn.Sequential(
    SinusoidalPosEmb(time_dim),       # 自定义模块也能放进去
    nn.Linear(time_dim, time_dim * 2),
    nn.SiLU(),
    nn.Linear(time_dim * 2, time_dim),
)
```

只要继承自 `nn.Module`，就能放进 `nn.Sequential`。

---

## 6. MLP 标准结构

```python
nn.Sequential(
    nn.Linear(d_in, d_hidden),
    nn.SiLU(),
    nn.Linear(d_hidden, d_out),
)
```

公式：**Linear → 激活 → Linear**，两层夹一个非线性激活是最基础的 MLP。

---

## 7. torch.arange —— 张量创建

| 方法                          | 说明                          | 示例                                |
| --------------------------- | --------------------------- | --------------------------------- |
| `torch.arange(n)`           | 创建 `[0, 1, ..., n-1]` 的一维张量 | `torch.arange(5)` → `[0,1,2,3,4]` |
| `torch.arange(n, device=d)` | 指定设备                        | `torch.arange(5, device="cuda")`  |

---

## 8. 张量形状操作

| 操作 | 说明 | 输出形状变化 |
|------|------|-------------|
| `t[:, None]` | 在第 1 维插入一维，变列向量 | `(B,)` → `(B, 1)` |
| `t[None, :]` | 在第 0 维插入一维，变行向量 | `(D,)` → `(1, D)` |
| `t.unsqueeze(0)` | 同 `[None, :]`，更显式的写法 | |
| `t.dim()` | 获取维度数 | |
| `t.float()` | 转为 float32 类型 | |
| `t.device` | 获取张量所在设备 | 返回 `cpu`/`cuda:0` 等 |

---

## 9. Broadcasting (Broadcasting)

不同形状的张量运算时，PyTorch 自动从最后一维向前对齐扩展：

```python
t = torch.tensor([1.0, 2.0, 3.0])   # (3,)
t = t[:, None]                       # (3, 1)
emb = torch.arange(4)                # (4,)
out = t * emb                        # (3, 1) × (4,) → (3, 4)  自动广播
```

规则：从右向左，维度相等或有一方为 1 即可广播。

---

## 10. torch.cat —— 张量拼接

```python
a = torch.tensor([[1, 2]])   # (1, 2)
b = torch.tensor([[3, 4]])   # (1, 2)
torch.cat([a, b], dim=0)     # (2, 2) 按行拼
torch.cat([a, b], dim=-1)    # (1, 4) 按列拼（最后一维）
```

`dim=-1` 表示沿最后一维拼接，是最常用的写法。

---

## 11. torch.数学运算

```python
torch.exp(x)     # e^x
torch.sin(x)     # sin(x)
torch.cos(x)     # cos(x)
```

均为逐元素运算。

---

## 12. F.pad —— 张量填充

```python
F.pad(x, (0, 1))         # 只在最后一维右侧补 1 个零
F.pad(x, (1, 0, 0, 1))   # (左, 右, 上, 下) 补零
```

`pad` 参数格式：从最后一维向前，每维两个数（左填充，右填充）。


## 13. F.interpolate —— 张量上/下采样

```python
F.interpolate(x, scale_factor=2, mode="nearest")
```

调整输入张量的空间尺寸（H、W），常用于 U-Net / DDPM 的上采样或下采样阶段。

| 参数 | 说明 |
|------|------|
| `size` | 目标输出尺寸 `(H, W)`，与 `scale_factor` 二选一 |
| `scale_factor` | 缩放因子，如 `2` 表示 H、W 各放大 2 倍 |
| `mode` | 插值算法：`"nearest"` / `"bilinear"` / `"bicubic"` 等 |

**常用 mode 对比：**

| mode | 原理 | 特点 |
|------|------|------|
| `"nearest"` | 新像素直接复制最近的原像素值 | 最快，输出有块状感，但**不引入伪影** |
| `"bilinear"` | 周围 2×2 像素的加权平均 | 平滑，但可能产生棋盘伪影 |

**生成模型（DDPM / U-Net）中的最佳实践：**

`Nearest 上采样 + 3×3 Conv` 是公认最干净的上采样模式：

```python
x = F.interpolate(x, scale_factor=2, mode="nearest")  # 粗糙放大，不引入伪影
x = self.conv(x)                                       # 3×3 卷积负责平滑+特征重组
```

- Nearest 只负责尺寸翻倍，不产生灰度渐变和棋盘伪影
- 后续卷积有可学习参数，自己学出最优平滑策略
- 比 `ConvTranspose2d` 和 `bilinear` 上采样更稳定

---

## 14. PyTorch 中最常见的装饰器

```python
@torch.no_grad()         # 禁用梯度计算（推理/验证时用）
def evaluate(model, data):
    ...

@torch.inference_mode()  # 更彻底的推理模式（PyTorch 1.9+）
def predict(model, x):
    ...
```

这两个不是装饰函数，而是**上下文管理器的装饰器写法**，和 `with torch.no_grad():` 等价。

---

## 15. nn.Conv2d —— 2D 卷积层

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0)
```

CNN / U-Net / DDPM 的核心算子，用可学习的卷积核在 2D 特征图上滑动做内积。

| 参数 | 含义 |
|------|------|
| `in_channels` | 输入通道数 |
| `out_channels` | 输出通道数（= 卷积核个数） |
| `kernel_size` | 卷积核尺寸，`3` 表示 3×3 |
| `stride` | 滑动步长，默认 1 |
| `padding` | 边缘零填充数，默认 0 |

**形状变化：** 输入 `(B, C_in, H, W)` → 输出 `(B, C_out, H', W')`

$$H' = \frac{H + 2 \times padding - kernel}{stride} + 1$$

**常用配置：**

| 配置 | 效果 |
|------|------|
| `kernel=3, padding=1` | **Same conv**，尺寸不变，只变换特征 — U-Net / ResNet 标配 |
| `kernel=1` | 1×1 卷积，只沿通道维做线性混合，不混合空间信息 |
| `kernel=3, stride=2, padding=1` | 下采样，H、W 减半 |
| `kernel=3, stride=1, padding=0` | 不做填充，每层尺寸缩小 2 |

**参数量：** $C_{out} \times C_{in} \times K \times K + C_{out}$（最后一项为 bias）。