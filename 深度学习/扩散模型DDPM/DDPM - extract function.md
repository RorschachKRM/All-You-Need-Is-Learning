---
title: extract辅助函数
tags:
  - DDPM
  - Diffusion-Model
  - Broadcasting
related:
  - "[[DDPM - Broadcasting]]"
---
# 简介
核心作用：从预计算好的 1 维调度数组（如 $\bar{\alpha}_t$ ）中，根据当前批次的时间步 t 取出对应的数值，然后把它变形（Reshape）成适合与图像张量 x 做元素级运算的形状。
即：从系数数组中按时间步索引提取对应值，并重塑为可广播到图像张量的形状。

`extract` 函数 = 索引器 + 形状变形器。

它把扁平的调度数组 `[T]` 变成可广播的 `[B, 1, 1, ...]`，从而让“一步加噪”和“反向采样”公式中的标量乘法能够直接、高效地作用于整批图像数据。


# 代码解释
## 代码1
假设我们有一个图像批次，形状为 `[B, C, H, W]`（比如 `[64, 3, 32, 32]`），`t` 是当前这批图像各自的时间步索引（形状为 `[B]`，比如 `[23, 45, 12, ...]`）。

（代码 1 来源：EM_deeplearning_beifen\DDPM\ddpm_model.py）
```python
def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:

    """Extract coefficients a[t] and reshape to (B, 1, 1, 1)."""
    bs = t.shape[0]
    # ensure a and t are on the same device
    a = a.to(t.device)
    out = a.gather(0, t)

    return out.view(bs, *((1,) * (len(x_shape) - 1)))
```
逐步解析：
1. **获取批次大小**
```python
bs = t.shape[0]  # bs = 64
```

2. **设备对齐（防止报错）**
```python
a = a.to(t.device)
```
确保调度数组 `a`（存在 CPU 或 GPU 上）和当前批次的时间步 `t` 在同一个计算设备上，否则 PyTorch 会报错。

3. **核心索引操作 `gather`**
```python
out = a.gather(0, t)
```
- `a` 是一个长度为 `T` 的一维数组（比如 `alphas_cumprod`）。
- `t` 是形状为 `[B]` 的整数索引数组。
- `gather(0, t)` 的作用是：**对于 `t` 中的每一个索引值，去 `a` 中取出对应位置的元素**
- 结果： `out` 的形状变成了 **`[B]`**。比如 `out = [a[23], a[45], a[12], ...]`。

4. **变形为广播形状**
```python
return out.view(bs, *((1,) * (len(x_shape) - 1)))
```
- 对于图像张量，`x_shape` 通常是 `(B, C, H, W)`，长度为 4。
- `len(x_shape) - 1 = 3`，`(1,) * 3` 生成了 `(1, 1, 1)`。
- `out.view(bs, 1, 1, 1)` 将形状从 `[B]` 变形为 **`[B, 1, 1, 1]`**。


## 为什么非要变形为 `[B, 1, 1, 1]`？

这是为了利用 PyTorch 的**广播机制（Broadcasting）**。
在 DDPM 的“一步到位”加噪公式中：
$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$
- x0 和 ϵ 的形状是 `[B, C, H, W]`。
- αˉt​​ 仅仅是一个标量系数，但 每个样本（批次中的每一张图）的时间步 t 可能不同。

如果我们直接拿 `[B]` 去乘 `[B, C, H, W]`，PyTorch 不知道如何将 `B` 维度的数值分配到 `C, H, W` 上。而当你把系数变成 `[B, 1, 1, 1]` 时：
- **第 1 维（B）**：对应批次中的每一张图，各自乘上自己的系数。
- **第 2、3、4 维（1, 1, 1）**：自动广播（复制）到这张图的每一个像素、每一个通道上。
这完美实现了“批次内每张图片拥有独立的时间步，但单张图片内所有像素共用同一个系数”的数学逻辑。


## 代码2

（代码 2 来源：\denoising-diffusion-pytorch-main\denoising_diffusion_pytorch\denoising_diffusion_pytorch.py)
```python
def extract(a, t, x_shape):
    b, *_ = t.shape          # 1. 获取批次大小
    out = a.gather(-1, t)    # 2. 索引取值
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))  # 3. 变形为 [B, 1, 1, 1]
```
- `b, *_ = t.shape`（Python 的解包语法，`*_` 吃掉后面所有维度）。
	- **分析**：因为 `t` 一定是 1 维张量（形状为 `[B]`），所以二者效果完全等价，纯属个人编码习惯。这种写法更“Pythonic”（简洁），但可读性稍差。

 - 索引维度 `dim=-1` vs `dim=0`（本质等价）
    - 上一版：`a.gather(0, t)`，明确在第 0 维（第一维）上索引。
    - 这一版：`a.gather(-1, t)`，在最后一维上索引。
    - 分析：因为 `a` 是一个 1 维张量（形状为 `[T]`），它唯一的维度既是第 0 维，也是最后一维（`-1`）。所以二者在数学上完全等价。写 `-1` 通常是一种习惯，表示“不论维度怎么变，我就在最后一维取数”，但在 1D 情况下没有区别。

- 变形函数 `reshape` vs `view`（安全系数不同）
    - 上一版：`return out.view(...)`。
    - 这一版：`return out.reshape(...)`。
    - 分析：
        - `view` 要求内存必须连续，否则会报错。
        - `reshape` 更“智能”，如果内存不连续，它会自动复制一份新的连续内存再变形，保证结果正确。
        - 在本场景中，`gather` 返回的新张量通常是不连续的，所以直接用 `view` 有时会踩坑，而 `reshape` 更安全。这一版的 `reshape` 比上一版的 `view` 更加稳健。