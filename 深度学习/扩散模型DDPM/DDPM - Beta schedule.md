---
title: 噪声调度、beta调度
tags:
  - DDPM
  - Diffusion-Model
related:
  - "[[DDPM - General Framework#3. 模块一：噪声调度（Noise Schedule）]]"
---
# 原理、目的
`make_beta_schedule` 里的各种 `if` 分支，就是同一个“造表工具”的不同“计算公式”。它们的任务都是输出一张 [ 1×T ] 的“噪声比例对照表”，后续所有的**前向加噪、反向采样算法**都只看这张表，不管它是怎么算出来的。

这组 $\beta_t$ 必须满足以下两个**绝对约束**，否则就不叫 DDPM 的调度器：

- **起点收敛**：$\beta_1$ 要足够小（接近 0），确保第一步图像几乎不变，信号保留率 $\bar{\alpha}_1 \approx 1$。
- **终点收敛**：累积到最后，必须满足 $\bar{\alpha}_T \approx 0$，也就是 $\prod_{t=1}^T (1-\beta_t) \approx 0$。这意味着经过 $T$ 步后，原始信号被彻底抹除，只剩下纯高斯噪声。
- **单调性**：$\beta_t$ 通常是递增的（或者至少是非递减的），因为越往后图像越模糊，加噪幅度理应越大或保持不变，不能忽大忽小（否则会造成训练不稳定）。

所有调度器都在暗地里追求一个目标：**让“信噪比（SNR）”在对数坐标下尽可能线性下降**。
- 因为人类感知和神经网络梯度对“倍数”更敏感，而不是“差值”。
- 线性调度的问题在于，它的 SNR 在中间段掉得太快（非线性暴跌）。
- 余弦调度、以及后来更先进的 sigmoid 调度、square（平方）调度，本质上都是在调整 SNR 下降的曲率，让模型在每一步遇到的“难题难度”尽量均匀。



在 DDPM 的前向过程（加噪阶段）中，加噪公式：
$$x_t=\sqrt{1-\beta_t}\cdot x_{t-1}+\sqrt{\beta_t}\cdot\epsilon$$

这里的 $\beta_t$（Beta）就是第 $t$ 步时加入的**高斯噪声的方差（加噪强度）**。
	 - $\beta_t$ 很小，说明这一步只往图片里加了一点点微弱的噪点，需要成千上万步才能变成纯噪声，训练和推理时间长得无法接受。
	 - $\beta_t$ 很大，说明这一步加了极其猛烈的浓雾噪声，图像会瞬间被噪声淹没，变成纯随机雪花，模型学不到中间过程的渐进变化。



# 具体调度策略
## Linear
### ：线性加噪，原始 DDPM 论文的做法
- 它生成了一个等差数列。让 $\beta_t$ 从极小的 `beta_start`（如 $0.0001$）匀速、直线地上升（**均匀递增**）到 `beta_end`（如 $0.02$）。
- 每一步增加的噪声量是恒定的（等差递增）。
- 问题：
	- **线性加噪“毁得太快了”**。当 $t$ 走到中间（大概 500 步左右）时，图片其实已经几乎变成纯白噪声了。这意味着后半段（500~1000步）加噪几乎是在做无用功，U-Net 在逆向去噪的早期阶段面临着极度缺乏信息的问题。
	- 在扩散的中间阶段，图像会退化得太快（信噪比骤降），导致模型在中间步很难学习。


（代码 1 来源：EM_deeplearning_beifen\DDPM\ddpm_model.py）
```python
def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02, schedule: str = "linear") -> torch.Tensor:

    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    ......
```

（代码 2 来源：\denoising-diffusion-pytorch-main\denoising_diffusion_pytorch\denoising_diffusion_pytorch.py)
```python
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02

    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)
```


## Cosine
### ：余弦加噪，Improved DDPM

- 它不再直接对 $\beta_t$ 进行插值，而是先设定了 $\bar{\alpha}_t$ （ `alphas_cumprod`，代表原图信号的保留比例），让原图的保留比例按照 $\cos$ 函数==平滑衰减==。最后反推出 $\beta_t$。
- 关键公式：$$\bar{\alpha}_t=\cos^2(\frac{t/T+s}{1+s}\cdot\frac\pi2)$$
	- 小常数 s=0.008

- 特点：
	- 它在两端（最初几步和最后几步）噪声增加得慢，在中间阶段增加得快。
- 好处：
	- 在扩散的前半段，图像能保留更多的清晰结构（不会烂得太快），让模型有充足的时间学习感知内容；在最后阶段平滑过渡到纯噪声。目前主流工程（如Stable Diffusion）更倾向于使用余弦调度或变体。

在扩散模型中，单步噪声方差 $\beta_t$ 和 累积信号保留率 $\bar{\alpha}_t$ 之间的核心数学关系公式是：

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$
（注：当 $t=1$ 时，$\beta_1 = 1 - \bar{\alpha}_1$）
### 详细推导过程

**第一步：定义基础关系**

在 DDPM 的前向过程中，设定了一个最基础的单步关系：
- **$\beta_t$**：单步加噪强度（方差）。
- **$\alpha_t$**：单步信号保留率。
它们的总和永远是 1，即：$$\alpha_t = 1 - \beta_t$$

**第二步：理解“累积”的含义**

$\bar{\alpha}_t$（alpha bar）代表的是从第 1 步到第 $t$ 步，**原图信号的累积保留比例**（原始图像信号在第 t 步时还剩多少比例。）。
它是历史所有单步保留率 $\alpha$ 的**连乘积**（Cumulative Product）：
$$\bar{\alpha}_t = \alpha_1 \cdot \alpha_2 \cdots \alpha_{t-1} \cdot \alpha_t$$

**第三步：建立递推关系**

根据上面连乘的定义，我们可以很自然地把第 $t$ 步的累积保留率，拆解为“上一步的累积保留率” $\times$ “当前步的单步保留率”：

$$\bar{\alpha}_t = \bar{\alpha}_{t-1} \cdot \alpha_t$$

**第四步：反推换算**

现在我们要逆向计算。把上面的式子变个形，求出单步保留率 $\alpha_t$：

$$\alpha_t = \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

最后，把第一步的基础定义（$\alpha_t = 1 - \beta_t$）代进去：

$$1 - \beta_t = \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$

稍微挪动一下位置，就得到了最终的结论：

$$\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}$$
- 重要一步加噪公式：$$x_t = \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon$$



（代码 1 来源：EM_deeplearning_beifen\DDPM\ddpm_model.py）
```python
def make_beta_schedule(T: int, beta_start: float = 1e-4, beta_end: float = 0.02, schedule: str = "linear") -> torch.Tensor:
	
	...
	elif schedule == "cosine":
		s = 0.008
		steps = T + 1
		x = torch.linspace(0, T, steps)
		alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
		alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
		betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
		
		return betas.clamp(1e-8, 0.999)
	else:
		raise ValueError(f"Unknown schedule: {schedule}")
```


（代码 2 来源：\denoising-diffusion-pytorch-main\denoising_diffusion_pytorch\denoising_diffusion_pytorch.py)
```python
def cosine_beta_schedule(timesteps, s = 0.008):
	
	steps = timesteps + 1
	t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
	alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
	alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
	betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
	
	return torch.clip(betas, 0, 0.999)
```

### 比较Linear、Cosine

- Linear 是最简单的调度策略，即直接线性增加所需的最终目标 $\beta_t$。
- Cosine 没有直接操作 $\beta_t$，而是操作与其有关的 $\bar{\alpha}_t$（alpha bar），算出 $\bar{\alpha}_t$ 的一个衰减序列，再通过公式换算，算出所需的最终目标递增的 $\beta_t$ 序列。
- 这样得到的 $\beta_t$ 序列虽然更繁琐，但是其质量比 Linear 直接得出的高。
### 为什么“质量更高”？——信噪比（SNR）的线性下降
这是 Cosine 调度最核心的物理优势。刚才提到的 $\bar{\alpha}_t$（信号残留率），决定了扩散过程中的**信噪比（SNR）**：$$\mathrm{SNR}=\frac{\bar{\alpha}_t}{1-\bar{\alpha}_t}$$
- **Linear 调度**：因为 $\beta_t$ 是直线上升，导致 SNR 在扩散的中间段（比如 $t=200$ 到 $800$）**发生断崖式暴跌**。图像瞬间变成纯雪花，模型在中间这几百步根本看不清任何结构，梯度近乎随机，白白浪费了计算资源。
- **Cosine 调度**：因为它是通过余弦函数精心设计了 $\bar{\alpha}_t$ 的下降速度，所以它的 SNR 在整个时间轴上**几乎呈现出完美的线性下降**。这意味着每一步，模型面临的”图片模糊程度”是均匀递增的，没有突然变瞎的阶段。

**结论**：Cosine 调度让模型在每一步都能学到“适量的信息”，训练收敛更快，最终的生成质量（FID分数）也更优。



## Sigmoid

（代码 1 来源：\denoising-diffusion-pytorch-main\denoising_diffusion_pytorch\denoising_diffusion_pytorch.py)
```python
def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return torch.clip(betas, 0, 0.999)
```