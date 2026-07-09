---
title: 推理/采样函数（DDPM 的反向去噪过程）、Algorithm 2 的代码实现
tags:
  - Sampling
  - DDPM
  - Inference
---
# 函数意义
sample_ddpm — DDPM 采样算法（核心推理逻辑）。
这是 DDPM 论文 Algorithm 2 的实现，从纯噪声迭代去噪生成图像。


# 代码1
（代码 1 来源：EM_deeplearning_beifen\DDPM\ddpm_model.py）
```python
@torch.no_grad()
def sample_ddpm(
    model: nn.Module,
    betas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    alphas_cumprod_prev: torch.Tensor,
    sqrt_recip_alphas: torch.Tensor,
    posterior_variance: torch.Tensor,
    class_onehot: torch.Tensor,
    az_vec: torch.Tensor,
    shape: Tuple[int, int, int, int],
    device: torch.device,
) -> torch.Tensor:

    """
    DDPM 条件模型采样循环。
    shape: (B, C, H, W)，即要生成图像的形状
    class_onehot: (B,10)， 即目标类别条件
    az_vec: (B,10)，即目标方位角条件
    Return x_0 到 [-1, 1] 之间的值（假设模型在该范围内训练）。
    """

    b, c, h, w = shape
    x = torch.randn(shape, device=device)
    T = betas.shape[0]

    for i in reversed(range(T)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        
        # model predicts noise
        eps_theta = model(x, t, class_onehot, az_vec)
        beta_t = extract(betas, t, x.shape)
        sqrt_recip_alpha_t = extract(sqrt_recip_alphas, t, x.shape)
        sqrt_one_minus_alphacum_t = extract(torch.sqrt(1.0 - alphas_cumprod), t, x.shape)

        # DDPM mean
        model_mean = sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_alphacum_t * eps_theta)

        if i > 0:
            noise = torch.randn_like(x)
            var_t = extract(posterior_variance, t, x.shape)
            x = model_mean + torch.sqrt(var_t) * noise
        else:
            x = model_mean
    return x
```

## 代码1详解

```python
@torch.no_grad()
```
`@torch.no_grad()` 是 PyTorch 的一个**装饰器（decorator）**，它的作用是：**在该函数内部禁用梯度计算**，等价于把整个函数体包在 `with torch.no_grad():` 里面。

`sample_ddpm` 是**推理/采样**函数（DDPM 的反向去噪过程），不是训练函数。
在推理阶段：
1. **不需要梯度**：采样只是前向计算，不需要反向传播，因此不需要构建计算图。
2. **节省显存**：不记录中间激活值（activations），显存占用大幅降低。
3. **加速计算**：PyTorch 不需要维护 autograd 计算图，推理速度更快。

这是 PyTorch 中的标准做法——所有 `forward` / `predict` / `sample` 这类==只做推理的函数==，都应该加上 `@torch.no_grad()`。


### 函数参数
```python
def sample_ddpm(model, betas, alphas_cumprod, alphas_cumprod_prev,
                sqrt_recip_alphas, posterior_variance,
                class_onehot, az_vec, shape, device):
```
`model` ： 去噪 UNet 模型。
	- **类型**: `nn.Module`（UNet 实例）
	- **作用**: 已训练好的噪声预测网络 $\epsilon_\theta(x_t, t, \text{class}, \text{azimuth})$。输入带噪图像 $x_t$、时间步 $t$、类别条件和方位角条件，输出**预测的噪声** $\hat{\epsilon}$。
`betas` ：噪声方差序列
	- **形状**: `(T,)`，默认 `(1000,)`
	- **作用**: $\beta_t$，前向扩散过程每一步添加的噪声方差。由 `make_beta_schedule` 生成，从 $\beta_1 \approx 10^{-4}$ 到 $\beta_T = 0.02$（线性或余弦调度）。
	- **采样中的用途**: 第 4 步提取 $\beta_t$ 参与反向均值计算。
`alphas_cumprod` ：累积 alpha 乘积
	- **形状**: `(T,)`
	- **定义**: $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s)$
	- **作用**: 表示经过 $t$ 步扩散后原始信号的保留比例。$\bar{\alpha}_T \approx 0$（几乎全是噪声）。
	- **采样中的用途**: 计算 $\sqrt{1 - \bar{\alpha}_t}$，用于反向均值公式的分母。
`alphas_cumprod_prev` ：前一步累积 alpha 乘积
	- **形状**: `(T,)`
	- **定义**: $\bar{\alpha}_{t-1}$，即 $\bar{\alpha}_t$ 左移一位，$\bar{\alpha}_0 = 1.0$。
	- **作用**: 用于计算后验方差 $\sigma_t^2$。
	- **公式**: $\sigma_t^2 = \beta_t \cdot \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$
`sqrt_recip_alphas` ：$\alpha_t$ 的平方根倒数
	- **形状**: `(T,)`
	- **定义**: $\frac{1}{\sqrt{\alpha_t}} = \frac{1}{\sqrt{1 - \beta_t}}$
	- **作用**: 反向均值公式的缩放系数。
	- **采样中的用途**: 均值公式的关键因子：$\mu_\theta = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta\right)$
`posterior_variance` ：后验方差 $\sigma_t^2$
	- **形状**: `(T,)`
	- **定义**: $\sigma_t^2 = \beta_t \cdot \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$
	- **作用**: 反向采样时加入的随机噪声的方差，控制每一步去噪的"随机程度"。
	- **采样中的用途**: 第 6 步 $x_{t-1} = \mu_\theta + \sigma_t \cdot z,\; z \sim \mathcal{N}(0,I)$。$t=0$ 时不加噪声。
`class_onehot` ：目标类别条件
	- **形状**: `(B, 10)`
	- **含义**: 10 类 SAR 目标（2S1, BMP2, BTR70, M1, M2, M35, M60, M548, T72, ZSU23）的 one-hot 编码。
	- **作用**: 告诉模型要生成哪一类目标。例如 `[1,0,0,...,0]` 表示生成 2S1 类图像。
`az_vec` ：目标方位角条件
	- **形状**: `(B, 10)`
	- **含义**: 方位角的 one-hot 编码（将 $0^\circ$–$360^\circ$ 离散化为 10 个区间）。
	- **作用**: 告诉模型要生成目标在哪个观察角度下的图像。
`shape` ：生成图像的形状
	- **类型**: `Tuple[int, int, int, int]`
	- **格式**: `(B, C, H, W)`，通常为 `(B, 1, 128, 128)`
	- **作用**: 确定初始噪声 $x_T$ 的形状和生成图像的尺寸。SAR 图像是单通道灰度图（C=1），分辨率 128×128。
`device` ：计算设备
	- **类型**: `torch.device`
	- **取值**: `"cuda"` 或 `"cpu"`
	- **作用**: 指定在 GPU 还是 CPU 上执行采样，所有张量都创建在这个设备上。



```python
    x = torch.randn(shape, device=device)  # 1. 从标准正态分布采样 x_T
    T = betas.shape[0]                     # 总步数 (默认 1000)
```

### 数学公式逻辑代码实现

```python
for i in reversed(range(T)):               # 2. 从 T-1 到 0 逐步去噪
    t = torch.full((b,), i, device=device, dtype=torch.long)  # 当前时间步

    # 3. 模型预测噪声
    eps_theta = model(x, t, class_onehot, az_vec)

    # 4. 提取当前步的系数
    beta_t = extract(betas, t, x.shape)
    sqrt_recip_alpha_t = extract(sqrt_recip_alphas, t, x.shape)
    sqrt_one_minus_alphacum_t = extract(torch.sqrt(1.0 - alphas_cumprod), t, x.shape)
```
提取参数（第4步）：

| 变量                          | 从哪张表提取                             | 数学含义                              | 在公式里的位置                                      |
| --------------------------- | ---------------------------------- | --------------------------------- | -------------------------------------------- |
| `beta_t`                    | `betas`                            | $\beta_t$，当前步的噪声方差                | 分子：$\beta_t$                                 |
| `sqrt_recip_alpha_t`        | `sqrt_recip_alphas`                | $\frac{1}{\sqrt{\alpha_t}}$，缩放系数  | 均值公式最外层：$\frac{1}{\sqrt{\alpha_t}}(\cdots)$  |
| `sqrt_one_minus_alphacum_t` | `torch.sqrt(1.0 - alphas_cumprod)` | $\sqrt{1-\bar{\alpha}_t}$，噪声系数的分母 | 分母：$\frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}$ |

> **`extract` 拿的是单个标量，不是序列。**  
> 
> `extract(a, t, x.shape)` = `a[t]` → reshape 为 `(B, 1, 1, 1)`。  
> 比如 `betas` 是 `(1000,)` 的完整表，$t=15$ 时只取第 15 步的**一个** $\beta_{15}$ 值，不是 $\beta_1$ 到 $\beta_{15}$
> 
> 因为 $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$ 这个累积连乘已在采样前通过 `build_diffusion()` 函数中的`torch.cumprod` **一次性预计算**完毕，存入了 `alphas_cumprod` 表。循环里 `extract(alphas_cumprod, t, ...)` 只是**查表取值**，无需重复计算。


```python
    # 5. 计算反向过程均值 μ_θ
    model_mean = sqrt_recip_alpha_t * (
        x - beta_t / sqrt_one_minus_alphacum_t * eps_theta
    )
```
**反向均值**（第 5 步）：
上一步的3个参数合起来就是 DDPM 反向均值公式：

$$\mu_\theta(x_t, t) = {\color{red}\frac{1}{\sqrt{\alpha_t}}}\left(x_t - \frac{\color{blue}\beta_t}{\color{green}\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right)$$

（红= `sqrt_recip_alpha_t`，蓝= `beta_t`，绿= `sqrt_one_minus_alphacum_t`）



```python
    # 6. 重参数化采样 x_{t-1}
    if i > 0:
        noise = torch.randn_like(x)         # 标准正态噪声
        var_t = extract(posterior_variance, t, x.shape)  # 后验方差 σ_t²
        x = model_mean + torch.sqrt(var_t) * noise       # x_{t-1} = μ + σ·ε
    else:
        x = model_mean                      # t=0 时不加噪声，直接输出均值
```
**重参数化采样**（第 6 步）：

$$x_{t-1} = \mu_\theta(x_t, t) + \sigma_t \cdot z,\quad z \sim \mathcal{N}(0, I)$$

其中后验方差：

$$\sigma_t^2 = \beta_t \cdot \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$$
**t=0 时不加噪声**: 最后一步直接输出均值（确定性），保证生成图像的清晰度。