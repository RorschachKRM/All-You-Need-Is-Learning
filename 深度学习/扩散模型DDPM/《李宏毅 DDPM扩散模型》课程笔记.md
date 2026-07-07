---
tags:
  - DDPM
  - Generative-Model
  - Deep-Learning
  - Diffusion-Model
related:
---

# 李宏毅 DDPM 扩散模型 — 课程笔记

---

## 一、自回归模型（Autoregressive Models）

自回归模型是思路最直观、在文本等领域统治力最强的一类。它的核心思想只有一句话：**把复杂的联合概率，拆成一串简单的条件概率，然后一个接一个地生成。**

假设我们要生成一个由多个部分组成的样本 $x=(x_1, x_2, \dots, x_n)$，比如一张图片的像素、一句文本的 token。自回归模型会利用链式法则，把联合概率分布 $P(x)$ 拆解为条件概率的连乘：

$$P(x) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1,x_2) \cdots P(x_n|x_1,\ldots,x_{n-1})$$

模型要学的，其实就是每一步"给定历史，预测下一个"的条件概率。

生成过程就变成了一个循环：

1. 模型根据已生成的部分，输出下一个部分的概率分布。
2. 从这个分布中采样出 $x_i$。
3. 把 $x_i$ 加入历史，继续预测 $x_{i+1}$，直到全部生成完毕。

### 自回归模型 vs. 扩散模型

| 维度 | 自回归模型 | 扩散模型 |
|------|-----------|----------|
| 生成过程 | 逐个元素（token/像素）顺序生成，不可并行 | 从噪声出发，对整个样本并行迭代去噪 |
| 速度 | 生成慢（串行），但训练可并行（Teacher Forcing） | 生成慢（需多步），但每一步并行处理全图 |
| 数据适用性 | 天然适合离散序列（文本、语音），也可做连续 | 天然适合连续数据（图像、视频），离散则需额外处理 |
| 概率建模 | 直接给出精确的似然估计 $P(x)$，显式密度模型 | 隐式，不直接给出 $P(x)$，通过去噪间接学习分布 |
| 主流应用 | NLP 大一统、代码生成、部分图像/音频 | 图像生成（Stable Diffusion）、视频、3D 等 |

---

## 二、Denoise Model（去噪模型）

**Reverse Process（逆向过程）**

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=M2Y5ZGQzZWM5MGRhNGU1NzgwZjY3YmI0MTVkODNjYzdfakJyQ2FoZjZFeTJ2VXdhTXlGYVQzb3FOTm1aODV5ajZfVG9rZW46VGZTMWJxbEk0b0JQUzZ4RkJKMWN5RkRzblVkXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=YjFkMzc3YzBkM2NkNjYyMjRjZjU1MzAwODdhOTdmZDlfSGN1TG1EZ0xTd05LWlJmYWllZDVkMmZzbEJucWswZGFfVG9rZW46TTd2M2JFN3ZDb294dHd4czZJbWNaWWRlblFnXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

---

## 三、训练扩散模型网络

**Forward Process（Diffusion Process / 前向加噪过程）**

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=N2MxYzhkN2E1NjhiOTI4ZTEzOWIyNmJhNzFhYmVhNzRfZEg5Y2NBc0tWdDVBS0tKZjU1dTliY294UG10VGJzbUNfVG9rZW46Rzc5MGJPdW1Mb2oydDd4Q0ltemNSQWo3bldiXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MmUzNTQwOTJlMTYzMjRmNDUyYTQwN2FjYzc1MTVjODNfRmxNeUVYb0RTT2NrRlc5ckNXT3oyOFBLRGJMRGpJck5fVG9rZW46VlBIZGJyVkkybzhFTzB4dE02bmNXajR6bmdlXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

---

## 四、文生图扩散模型

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=ZjY4OTA4OGNhNGQ2M2QzY2U4NTJhMTljNWExN2NmNDlfY05yV3k5akZoV25sNUlXQ0lGMXhHNTR4Sm96cE9KTkhfVG9rZW46QlVEMGIzQkdnb3loY1R4UkIwOWN3bkxwbnBjXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDg1YmM3MWI5Y2E3YTk0ZmJjNWY1YzE3NDUxMDgzNzFfWWxZdHdNbzVvaFVCajdTTnVZdzlIUWVXODF4SWlvaHhfVG9rZW46UWw3SWJFTXVhb0RnUmV4b1lMaWMybFFmbmtjXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MjBjZGEwOTIwMGI1ZGJmNDkzOTRlY2E3MjQ1NDZiNzdfTGNmYmRTNzd2THoyQWh0bVdnd1BtcmxQOTk3emIwbkJfVG9rZW46Uk1kaWJGenA0bzdrQUF4OTRaV2NPMnZmblNnXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

---

## 五、文生图模型框架

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NjJmZDgzMTVlOTA5Y2FkODM0MDBmZDkzMmM5ZmM1YjJfNkFBQ3NoME12YzVhZjdXdmlKNWNXdlVqZ3dObGVyMFZfVG9rZW46QVlVdGJQWVAyb0pKUnN4YUpnVmNWS2FLblVaXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

### 5.1 Text Encoder 部分

#### FID（Fréchet Inception Distance）

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NzNhZTA0ZGEyZTkzZTg1ZWZjZDNhNjA2OTEwZGZiNGJfVEQ0RXduUjQzdVpDaE1UNWJNZ0xSNkpSbERxeEpjdW1fVG9rZW46RFk0U2JsamI2b2pJcnd4WlRxVGNlUHJSbnVkXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

评估生成模型（如 GAN、Stable Diffusion 等）生成图像质量的常用指标。它通过计算**真实图像与生成图像**在**高层语义特征空间中的分布距离**，来衡量**生成图像的质量和多样性**。（与传统的均方误差 MSE 等指标不同，FID 考虑了图像特征的分布，而不仅仅是像素值的差异。）

**核心思想**：理想的生成模型应当让生成图像的**分布**尽可能接近真实图像的分布。

FID 并不直接比较像素，而是将两类图像送入**预训练的 Inception v3 网络**，提取出**包含语义信息的特征向量**，然后**假设**这些**特征服从多元高斯分布**，最后计算两个**高斯分布之间**的 **Fréchet 距离**（也叫 Wasserstein-2 距离）。

> **距离越小（FID 值越小）**，表示生成图像的整体统计特性越接近真实图像，意味着图像更真实且模式更多样。

**计算步骤**：

1. **准备样本**：收集一组真实图像（如训练集）和一组待评估的生成图像，通常要求每类不少于 10,000 张以保证统计稳定性。
2. **提取特征**：用 ImageNet 预训练的 Inception v3 模型，取最后一个池化层（pool3）输出的 2048 维向量作为每张图的特征表示。
3. **拟合高斯分布**：对真实图像特征计算均值 $\mu_r$ 和协方差矩阵 $\Sigma_r$；对生成图像特征计算均值 $\mu_g$ 和协方差矩阵 $\Sigma_g$。
4. **计算 Fréchet 距离**：

$$\mathrm{FID} = \|\mu_r - \mu_g\|^2 + \mathrm{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{\frac{1}{2}}\right)$$

其中 $\mathrm{Tr}$ 是矩阵的迹，$(\Sigma_r \Sigma_g)^{1/2}$ 是两协方差矩阵乘积的矩阵平方根（计算时常用 SVD 分解）。

**指标解读**：

- FID 越小 → 生成质量越高、多样性越好。如果一个模型只生成了少数几类逼真的图片（模式坍塌），那么生成特征的协方差会很小，与真实协方差差异变大，FID 会升高。
- 典型参考值：
    - 50+：图像模糊或严重模式坍塌。
    - 10～20：较好的生成质量。
    - <5：几乎与真实图像难以区分（需注意过拟合风险）。

> 现在的 AIGC 论文基本都将 FID 作为必须报告的核心指标，IS（Inception Score）仅作辅助。

#### CLIP（Contrastive Language-Image Pretraining）

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MDAzMmI1MzFjY2M0ZGY4M2VkZmJjODhmNzQ2ODZlMjlfTTd3ZG9Gdldqb1R5d0FRbG9ZSm5OUmQxUnBmVWZnWUxfVG9rZW46RmJXZWJqOWVxb0tMWWx4M1NNVmMxMXBSbmdmXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

**CLIP 模型**：核心能力是能把**文字和图片**映射到**同一个向量空间**里，直接判断一段文字和一张图片有多匹配。

传统模型需要为每个图片分类别训练，而 CLIP 学到的是一种**通用的"图文对齐"能力**。

- **训练方式**：它用 4 亿对从网上爬取的"图片-文字描述"进行对比学习。模型要做的，是从一堆图片里找出哪张图对应给定的文字，反之亦然。
- **结构**：一个文本编码器 + 一个图像编码器。它们输出的向量可以直接计算余弦相似度（cosine similarity）。
- **妙处**：训练完后，它可以直接用来做"零样本分类"——比如你给它一张猫的图，它不需要事先见过"猫"这个类别名，只要把图与"一只猫""一只狗"等文字分别算相似度，最高的就是答案。

这就是很多 AI 绘图工具（Stable Diffusion、DALL·E 等）的底层能力，也是它们能理解复杂提示词的关键。

**CLIP Score（图文一致性分数）**：FID 只看图像本身的真实性，没法判断生成图是否遵从了文本指令。而 CLIP 指标补上了这个缺口。

这是最直接的指标：把生成的图片和输入的提示词一起送入 CLIP 模型，计算它们的余弦相似度。

- 分数越高 → 生成的图片越贴合文字描述。
- 常用于评估文本到图像模型（如 Stable Diffusion）的指令遵循度。

> 注：因为计算简单直观，它成了文生图模型必报的指标之一，但缺点是可能对"刷榜"敏感（生成简单背景的突出主体容易拿高分），需要配合其他指标。

### 5.2 Decoder 部分

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=ZGI0MTQ0YjJkY2MyOTNmYzg0MTQyMTQ1NWE2ZDhjNTNfck9zUHppZllydlNTQTNmOG03T1dTTVlwOWxsN0h3Q09fVG9rZW46T3phTWJ4amgwb1BzalR4SXNJWmM5RDZmbm5lXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NTI5NmM1ODU2ZjA5M2M4NTkxZTgyOWZiZTFhN2JmZjlfZkpaMTU2Z3BvdGZqQlh0YkVPczVJR1BFbEtCRXplTWVfVG9rZW46U3FjdWJCY2ljb0hUb1h4Smk5SmNCWnNXbm5nXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=YTdkNTZkMWQ0MDRjM2MyNWYwYTZhZTI2NTRiNzViODFfZ3MzM29BZUFFa1g1U2tzSXMwQTVNYjZ2S0JXR3JvNW1fVG9rZW46VFBTQWJxUThtb3pOQ014cktsNGM2bGFtbkVmXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

### 5.3 Generation Model 部分

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MGY3NzNhYjA4ZTU3MzRlZjdlYzZlYzM2ZGQ1NGFjMDFfZVFjaTZQbHFMN2FXcXBwVWZiV2xxSTJrbWhONUM3aHpfVG9rZW46R0xJWWJzMVJ3b2N5SnF4d1dMNmNiWUZpbmllXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

**生成模块（Denoise）**：

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=YTI5YTM1OTA5ZDNiMmE3Yjg3NWEzNDc0M2UzODcwNDZfSTJ6dHkwd3JCdzUwcmhZY1E2Y3JSMlQ5ZTYwQjVidWhfVG9rZW46Rlo2a2JZTVhzb3dUWjR4VXNzaWNEMnE1bnRoXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

---

## 六、Latent Diffusion Model（LDM，潜在扩散模型）

是对标准扩散模型（如 DDPM）的一项关键改进：它不直接在像素空间里做加噪和去噪，而是把图片压缩到一个极小的**"潜在空间"（Latent Space）**里，在这个压缩后的低维空间里跑扩散模型。

### 6.1 为什么需要 LDM：像素空间扩散太贵了

标准 DDPM 在**原始像素空间**上工作。一张 512×512 的图片有 262,144 个像素点，每个像素又有 3 个颜色通道。在这个高维空间里反复做前向加噪、反向去噪，计算量极其庞大，训练和生成都非常缓慢，难以扩展到高分辨率。

LDM 的核心洞察是：**图像的语义信息**（比如物体是什么、布局如何）其实可以用远少于像素数量的维度来表达。如果把扩散过程压缩到一个"压缩过的语义空间"里进行，就能大幅降低计算开销。

### 6.2 步骤

LDM 在标准扩散模型的基础上，增加了一个"感知压缩"的外壳：

| 组件 | 功能 | 空间尺寸变化示例 |
|------|------|------------------|
| 自编码器（VAE） | 编码器 $E$ 把图像 $x$ 压缩成潜在表示 $z = E(x)$；解码器 $D$ 把 $z$ 恢复成图像 $\tilde{x} = D(z)$ | 512×512 → 64×64×4（压缩 8 倍） |
| 扩散模型（U-Net） | 在潜在空间 $z$ 上进行标准的加噪、去噪过程，并且可以接收文本、图像等条件 | 在 64×64 的潜在空间上运行 |
| 条件注入机制 | 用交叉注意力把文本/其他条件注入 U-Net，控制生成内容 | 与标准条件 U-Net 一致 |

训练分两个阶段：

1. **训练自编码器**：让 VAE 能把图像压缩得很好，同时解码还原也很逼真（这步与扩散过程无关）。
2. **训练潜在扩散模型**：在冻结的 VAE 编码器的潜在空间里，训练扩散模型去生成 $z_0$。条件（如文本）通过交叉注意力在这个阶段注入。

生成时：采样一个随机潜在噪声 $z_T$ → 用条件 U-Net 逐步去噪得到 $z_0$ → 通过 VAE 解码器还原为像素图像。

### 6.3 优点

把扩散过程搬到潜在空间，带来了根本性的好处：

- **计算量与训练速度大幅降低**：潜在空间的维度通常只有像素空间的 1/48 甚至更小，扩散模型的计算量呈指数级下降。这让在消费级 GPU 上训练高分辨率生成模型变为现实。
- **生成质量不降反升**：压缩过程像一个"语义过滤器"，会丢掉像素级别的无关细节（比如单个像素的微小波动），保留主要的语义信息。扩散模型因此能更专注于学习图像的内容与结构，生成的大体布局和物体形态更合理。
- **天然支持条件控制**：在这个压缩后的低维紧凑空间里，文本、布局、深度图等条件信号能更直接地影响全局构图。这也是 Stable Diffusion 等模型能实现强大文生图能力的关键前提。

---

## 七、主流文生图扩散模型

### Stable Diffusion

是一种 Latent Diffusion Model（LDM，潜在扩散模型）。

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NWEyNDhiNWExMWQyMjhkMjcxM2Y5NDFmYjc3ODZkMGFfeXRwcDROb1JPWTMxVk9RbWRVN29aOHpTTlNsZDgzWGVfVG9rZW46UUtxRWJ3WDA5b1JZVUp4a3NMbmNYTkdvbkVkXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

### DALL·E 系列

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NGM3ODMxZGQ4ODYyYzQwZDE0YWRiNDFjODhjNDU0NjhfTnZtMHlXMm9nY3dtbGZoaXdGbHRIMW9KdmVUVGplaDRfVG9rZW46SzZsS2J0eGpWb3JTMzZ4aU1GUWNBWE9pbjNmXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

### Imagen

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NzRjMzdlZDBkMDAzMWJkZDczNzllNjZhYmI2ZGEwMmRfUWZaYkdIcE50cVQwNmtUdUE3cUlCejczNFBQd0MzMVNfVG9rZW46THZSdmJLRWJJb0R6MXZ4MjlLbWNVYTNRbm5oXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

---

## 八、VAE vs. DM

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MmE4ODY5NTczNGJkNTg3NjRiOTI1MjkyZTQ5M2ZiZTZfRmhEejN3M2lqclp5bDFkWWdZZ1hmVmpVSlRzOUVWUVhfVG9rZW46RTc3MmI4SjhQb3U5NU54TU5JQWNzUjNnbmFmXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

---

## 九、DDPM 算法原理详解

### 9.1 训练算法

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=ZTMxZTZjNDlkODRkZjIwNzVkOTdjMWRhN2FmN2VkYmVfRVV6ZndDdVpGanhmRnRla0FMdG00SlA1MnduQ0dzcHNfVG9rZW46WEV4WGJreHV0bzJvTTR4N0tTMmNBNWxJbjFnXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

噪声是一次性加入的，噪声预测也是要一次性预测出来：

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MTVlNmYxYTdjNTA3NWQxYjVlZTM4MjFjYTFmMGZhNGZfcVh2Q2tVYUMxSk43T21oQ2lWQ3p3a0JkVVYyNnB3VmhfVG9rZW46WDJKamJFbTlJb0wxR2J4QklLaWN4a3ZzbkxmXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

### 9.2 去噪算法（采样 = 生成）

在扩散模型（DDPM）的语境中，"采样"与"生成"本质上是同一件事，但使用"采样"一词更精确地反映了其概率模型的本质。

扩散模型是一个隐变量生成模型，它定义了一个关于数据 $x_0$ 的分布 $p_\theta(x_0)$。要生成一张新图像，就需要从这个分布中随机抽取一个样本（Sample）。这个过程在统计学和机器学习中被称为从模型分布中采样。

- **生成（Generation）**：更偏重任务目标（得到一张看起来真实的图像）。
- **采样（Sampling）**：更偏重数学操作（从概率分布中抽取一个点）。

Algorithm 2 的每一步也都在做"采样"：先是从先验分布 $p(x_T) = \mathcal{N}(0, I)$ 中采样 $x_T$，然后逐步从条件分布 $p_\theta(x_{t-1} \mid x_t)$ 中采样 $x_{t-1}$。所以整个链条就是一系列从分布中采样的过程。

- 在 **VAE** 和 **GAN** 中，通常使用"生成"一词，因为 VAE 有解码器，GAN 有生成器，这些网络是确定性映射（除输入噪声外）。
- 在**扩散模型**和**基于能量的模型（EBM）**中，更常用"采样"，因为它们依赖迭代随机过程（马尔可夫链、朗之万动力学）来产生样本，每一步都包含随机性，不是简单的单向前馈。

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NTRjN2M4YzY2NWY1MzA2OThjMTZhOGQwMzgzYmMxYTdfT0xnVklYWXdEbXo2SEZNcWdyOGViUUdIS3dwTnU3OW1fVG9rZW46UGRTUGJGS21Zb1RBWmJ4QjFoNGNtdWp3bklnXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NjY4NjhiOTk1Yzk0NjEzMDI5OTA3YTljMDRlNTM0MDBfZHFjdFgwRlpsbWk4bk1JMkF2R2xmUzNsdXRZTFhycUxfVG9rZW46S2ozOWJGVGl4bzNodnR4TXB1VGM4eXFvblhnXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

**为什么去噪时还要再加一个噪声 $z$？**

去噪时加噪声 $z$，是为了用可控的随机性，去模拟并逆转加噪时那不可知的随机性，从而让模型能生成一个充满不确定性的、鲜活的分布，而非一张呆板的图像。

### 9.3 确定网络参数：最大似然估计（Maximum Likelihood Estimation）

DDPM 不是直接做最大似然估计，而是通过优化**变分下界（ELBO）**来近似最大化似然。

假设我们有一堆真实图片样本，**生成模型的目标**永远是：让模型生成这些样本的概率最大。这就是最大似然估计的核心。

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=ZDA1ZWZiOWE2OTIxNDBiNzZlZGM4MDhkMDJmZjM3YjFfM29XVHlDRTFZVWNMbGN0ZnFXWno1bXJyME1ZNVNMSUhfVG9rZW46TDFFamJzdlVIb0RPOGZ4NmdhQWNDNHNablNoXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

- **$x_0$**：代表一张"具体的真实图片"。比如你的训练集里有一张非常可爱的橘猫照片，在数学上就是一个特定的像素矩阵。
- **$\theta$**：代表"神经网络的参数"。也就是 U-Net 脑子里的那几千万个权重（神经元连接的强度）。训练开始前这些参数是随机的，训练过程中我们就是在不断修改这个 $\theta$。
- **$P_\theta(x_0)$**：**在当前神经网络参数为 $\theta$ 的情况下**，如果我们给模型输入一堆纯随机的噪声，让它自由发挥去"去噪"生成图片，**它最终生成的图片，完完全全、一像素不差地就是那张橘猫照片（$x_0$）的概率有多大。**

> **"最大化似然（Maximum Likelihood）"的真面目**：调整参数 $\theta$，让模型生成真实世界数据 $x_0$ 的可能性达到最高。

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=ZWRhZmRiZTNmMDkyYTc4ZDc3YzE4OWRmYWRjZjUxMDBfcnBNNExDTTlUWTBuTGNIUUo5ZjB2M2FUNk5FV2dBTFlfVG9rZW46VGxRWWJhSnVFb24yaFR4OEtGc2NwM2I0bjFkXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=YjcwOTE0NGM3NzNiOTM1ZDE5OWUxYzdkODgyYjJlMjBfYWRwejIya2Q0VEdza1hBWGN6R2FxZkFCclg5YVBTbVNfVG9rZW46UWVtR2JiNnB1b1paeUp4S2huU2NKbXlsbjZlXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

对扩散模型来说，我们想最大化模型给出的数据似然：$p_\theta(x_0)$。

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=OTM1ZDQ0OTU5YjlmOGFlZjA2ZTAyNGYwOTgzYWM3MTdfRTZGWW82NnlJYlNwYWlkd2xYV2ZuWVhuMG5KbjBOUUNfVG9rZW46V1h6RmJ2WFdxb3JycEx4WjdyTGNYQzBEbmZkXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

- $P_\theta(x)$ 表示该分布是通过 $\theta$ 所定的，$\theta$ 是网络里面的参数（一个 denoise 过程是通过 $\theta$ 决定的）。
- $P(x_T)$ 与 denoise 过程无关，就与 $\theta$ 参数无关，是一开始就采样的。

**变分推断**的做法（具体推导同 VAE 方法）是：引入一个已知的、容易计算的后验分布 $q(x_{1:T} \mid x_0)$，也就是我们设定的正向加噪过程。利用它，我们可以推导出**对数似然的一个下界（ELBO**，Evidence Lower Bound）：

$$\log p_\theta(x_0) \geq \text{ELBO} = \mathbb{E}_q\left[\log p_\theta(x_{0:T}) - \log q(x_{1:T} \mid x_0)\right]$$

**最大化这个 ELBO**，就相当于在"抬高地板"，间接推高我们真正关心的对数似然。

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NmNlODk0Yzg2NzJkNWE3MWIwNDA1ZjRhODdlNjk2Y2FfU0RFS0RnaUxkRUdDZml0ZzVoOFJvSWtNRmdCc0U4bzFfVG9rZW46UUhlQWJhNXRkb0NIcHR4VFZQdmNrcUpibmtmXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

经过推导，这个 ELBO：

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NmZhMGNjMDg2NWNmZDUyZGNmNDQ0MzA4ODk5NWNjYTBfUG9ZTTJzM1FuVTZYRFBUTEJMTkVlVVd4RkJhRldyRGxfVG9rZW46UmdkRGJXSDJMb2U5Y2l4Y0JmUGNjV283bk5iXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

可以拆成一系列 KL 散度之和，最后化简为：

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MTkxODJhMWRlNjg5Zjk3N2M2ZTBjMmE0M2EwN2VhNjBfTFhnM095Qm01cEd2RzFnakJKZU9CaXhrd1VHRk9ERXdfVG9rZW46WGxoWmI3QkVEb0psZGt4bFdINmNlajdEbm9mXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

继续推导以上所得结果，可知所求 $q(x_{t-1} \mid x_t, x_0)$ **依然是高斯分布**：

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MmVhMWJjZGMzNjc4NDhhMzA5MDU3MTA4NDAyMzZlYTdfdlhnYU82U3JjVGdodFhOQllrenlJVnNYcHI4NFJvVlBfVG9rZW46R3NHQWJGcElrb0pMZ2t4NUx2T2NqZmk0bnFJXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

**如何 minimize 这个 KL divergence：**

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NjE0ZDUwYmM0ZThiNmFkZjMzYjc2NjQyOTFjZDhiM2ZfWnVacEF6aEExeVhTV1JiNmR6VVFrT2pxR2NRS3RZZlRfVG9rZW46T2M0U2JpTDQ0b2Ewd3l4UDRxQmNZVks1bm9nXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

在 DDPM 的设定里，**真实分布 $q$ 和预测分布 $p_\theta$ 全都是高斯（正态）分布**，并且作者把它们的方差设定成了固定的常数。在数学中，如果两个高斯分布的方差一样，那么计算它们之间的 KL 散度，**就直接等于它们"均值"之间的欧几里得距离的平方（也就是 MSE）**。即：

$$KL(q \parallel p_\theta) \propto \| \mu_q - \mu_\theta \|^2$$

概率论问题，被成功降维成了一个找中心点（均值）的几何距离问题。

在严格的数学推导下，最小化 KL 散度推导出来的 MSE，前面是**带有一长串时间步 $t$ 相关的权重系数的**。

但是论文作者在做代码实验时发现了一个反直觉的现象：**如果把前面那堆复杂的权重系数直接砍掉，把所有的权重都强行变成 1**（也就是论文里的 $L_{simple}$ 简化版损失函数），模型不仅学得更快，而且生成的图片质量竟然更好了！

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=YTEwMjc4ZWQxNzU2YTQxNGYyYTQ4ZTIyZDM4MjJjZDVfRE1KSjN5cnZiaWd1djJTZGtzenRjVk1pNUZjUnRBNzBfVG9rZW46TjZqUWJUcU1Nb0IzM0d4NjBZTWNGTHZXbnFoXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

此式子就与论文中算法相同：

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=YThmNTRhODBhNGI5MGEyNjIxMGI5NzUxMjBlOWQzOTRfUVZjdElIbndUb25kTWU2MnVaTGFXajNxSEZWN0R2THNfVG9rZW46VlJtRWI5Q2Znb0s1eWR4TDluNmNDMUthbmtnXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

**为什么以上算法结果中还需要再加入一个噪声项 $\sigma_t z$？**

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MTI0ZTFkNWEzNGM2NmVlYWRlYWM3OWExMGZlMmJkM2ZfMGQydmQyVWdGOG9rMW13YlBHS0VTanlxTEdwR1Jjd3ZfVG9rZW46WVEzYWJ3SWxTbzJMVmF4SkdXUmNLZ2lSbnlnXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=OGE3ZDVhNmFkMzVhNzRjNWE2NzhmMWU4OGVkOTg1MGJfVXVUeXpoaXJWTnU0NFJtVUlSZVVWZlRHSkpINE9CVHpfVG9rZW46S1NYNmJ5RWw1b0hiTEl4cTVFcmNkNGRZbkdoXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

**同性质问题：为什么大语言模型如 GPT 在生成文句的时候需要 sample，而不是直接取概率最大的？**

为了通过引入**受控的随机性**，来保证生成结果的多样性和真实性。

**贪心解码（直接取最大）的灾难：退化与重复**

如果每一步都选 argmax 概率最高的 token，生成过程就变成完全确定的。这在实践中会导致：

- **高频循环重复**："我今天很开心，因为我很开心，因为我很开心……" 语言模型一旦进入一个高概率的短语或句式，就会不断自我强化，陷入死循环。这叫"退化"。
- **内容平淡、陈词滥调**：模型永远选它训练时见过最频繁、最安全的词汇组合，生成的东西像官方套话，毫无新意。
- **无法展开创造**：类比、幽默、诗意，这些往往需要从非最高概率的词里"冒险"选择。贪心解码扼杀了所有可能性。

从原理上看，**单步贪心最优不等于整体序列最优**，但这里的问题比你想的更严重——即使我们用束搜索去近似全局最优，在开放式文本生成中，束搜索同样会产生重复和无聊的句子，因为语言的概率分布有非常强的长尾和胖尾特性，最高概率路径往往是一条密度虽高但极其枯燥的"平坦大道"。

**采样的作用：模拟人类语言的随机性与多样性**

人类在说话写作时，每个词的选择本身就是一个概率分布里的采样。同一个意思，有无数种表达。因此，从模型预测的分布 $P(\text{token} \mid \text{context})$ 中随机采样，能更真实地还原这种多样性。

- 打破重复循环：哪怕概率最高的词有 40%，我们仍有 60% 的机会选其他词，从而跳出不断自我复制的陷阱。
- 产生信息量：低概率词往往携带更多语义信息（"猫"比"的"概率低，但说出"猫"后对话会走向更具体的场景）。
- 创造惊喜：GPT 令人惊艳的"理解力"和"创造力"，很大程度来自采样带来的随机跳跃，而非一味走最确定的路径。

直观类比：就像 DDPM 去噪时加 $z$，它让模型能**从同一个初始噪声产生不同图像**；语言模型采样，也让同一个开头能走向千万种续写。

| 维度 | DDPM 逆向去噪 | GPT 文本生成 |
|------|-------------|-------------|
| 基本操作 | 预测均值 $\mu$，再采样加 $z$ | 预测 logits 分布，再采样 token |
| 随机性的作用 | 保证图像多样性，修正偏差 | 保证文本多样性，跳出重复 |
| 去掉随机性（$\sigma=0$ / argmax） | 变成确定性的 DDIM，生成单一 | 变成贪心解码，生成枯燥重复 |
| 控制随机性的阀门 | 方差调度表 $\beta_t$ | 温度 $\tau$、Top-k、Top-p |
| 理论根源 | 逆向过程被建模为马尔可夫链上的概率分布 | 语言被建模为序列上的概率分布 |

### 9.4 重参数技巧：如何一步加噪得到 $q(x_t \mid x_0)$

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=MTA2OTU2MzRhY2ZiOWM1MDk2ZWJiNGM3NGZmZjAxNTBfVmt2R2g1RVBQUTU0R0Q3Sks5UU53Rkt0SnNpZlhpcWdfVG9rZW46RlYzVGJuYkI4b3VFdmV4T0FCY2M3dHdpbjViXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=ZmMyN2ZhNTlmMzdjZTk2MzgwODY1YmNkNzE5YWNlNWNfQXRsWHcwTzk4ZGMwOGxPckZVdUhmcG1aUkRTSjZ4dk5fVG9rZW46QWNpamJacDlRb0g4WER4SU1YVGMxWHJvbkdlXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)

![](https://bcn31a22ghof.feishu.cn/space/api/box/stream/download/asynccode/?code=NzU1ZDVhZmQ0MDg1NWEzZWE0MTJmMGNiYTE2MmUyMjhfV3A5a3dZeUh0MGdJd1VtSk55dGpza1k1YkxiWXFPZG5fVG9rZW46Q3N4MmJIV3pXbzhHVFp4aWFTVmNxT0dibm1lXzE3ODI4NzgwMzk6MTc4Mjg4MTYzOV9WNA&add_watermark=true&scene_type=CCM)
