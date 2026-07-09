---
title: 采样、推理（inference）、生成（generate）
tags:
  - DDPM
  - Sampling
  - Inference
  - Generate
related:
  - "[[DDPM - General Framework#8. 模块六：采样管线（Sampling Pipeline）]]"
---
# 代码1
（代码1来源：EM_deeplearning_beifen\DDPM\generate.py）

## build_diffusion()
与 `train_ddpm.py` 中的函数相同，但只计算推理需要的系数（不需要 `sqrt_alphas_cumprod` 和 `sqrt_one_minus_alphas_cumprod`）。


## main()：推理主函数
整个 `main()` 函数是一个**用训练好的条件 DDPM 模型生成图像的推理脚本**

```
# 整体数据流总结
测试txt文件 → SAMPLE_Dataset → DataLoader
                                ↓
                         (labels, az, names)
                                ↓
              ┌─────────────────┼─────────────────┐
              ↓                                    ↓
     class one-hot (B,10)              azimuth Fourier (B,10)
              └─────────────────┬─────────────────┘
                                ↓
                   ConditionalUNet + DDPM采样
                                ↓
                       生成图像 (B,1,128,128)
                                ↓
                      按类别分文件夹保存
```

### 命令行参数解析
```python
parser = argparse.ArgumentParser(description="...")
parser.add_argument("--txt_file", ...)     # 测试集 txt 文件路径
parser.add_argument("--batch_size", ...)   # 批次大小，默认32
parser.add_argument("--ckpt", ...)         # 训练好的模型权重路径
parser.add_argument("--save_dir", ...)     # 生成图像保存目录
parser.add_argument("--timesteps", ...)    # 扩散步数，默认1000
args = parser.parse_args()
```
定义了 5 个可配置参数。`--txt_file` 指向测试集文件（格式为 `路径 标签`），`--ckpt` 指向训练好的 UNet 权重，`--save_dir` 指定输出目录。


### 环境准备
```python
os.makedirs(args.save_dir, exist_ok=True)        # 创建保存目录
device = torch.device("cuda" if ... else "cpu")   # 自动选择 GPU/CPU

data_transforms = transforms.Compose([transforms.ToTensor()])  # 只做 ToTensor，无归一化
dataset_test = sample_dataset.SAMPLE_Dataset(...)               # 加载测试集
test_loader = DataLoader(dataset_test, batch_size=..., shuffle=False)
```
关键点：
- `transform` 只有 `ToTensor()`，没有 `Normalize`——因为 DDPM 在 `[-1, 1]` 范围工作，后续生成后再手动映射到 `[0, 1]`
- `shuffle=False` 保证每次生成的顺序一致，方便对照


### 模型加载
```python
model = ConditionalUNet(img_channels=1, base_ch=64, time_dim=256, cond_dim=64).to(device)

state_dict = torch.load(args.ckpt, map_location=device)
model.load_state_dict(state_dict)
model.eval()
```
- `img_channels=1` → 灰度图（单通道）
- `base_ch=64` → 基础通道数
- `time_dim=256` → 时间步嵌入维度（用于告诉模型当前是第几步去噪）
- `cond_dim=64` → 条件嵌入维度（类别 + 方位角的联合编码）
- `model.eval()` 关闭 Dropout/BatchNorm 的训练行为


### 扩散系数构建
```python
    diffusion = build_diffusion(args.timesteps, device)
    betas = diffusion["betas"]
    alphas_cumprod = diffusion["alphas_cumprod"]
    alphas_cumprod_prev = diffusion["alphas_cumprod_prev"]
    sqrt_recip_alphas = diffusion["sqrt_recip_alphas"]
    posterior_variance = diffusion["posterior_variance"]
```
`build_diffusion` 函数计算 DDPM 所需的全部系数：

| 变量                    | 公式                    | 含义                                   |
| --------------------- | --------------------- | ------------------------------------ |
| `betas`               | 由 schedule 定义         | 每一步加的噪声方差                            |
| `alphas_cumprod`      | `cumprod(1 - betas)`  | 从 t=0 到当前步的信号保留比例 ($\bar{\alpha}_t$) |
| `alphas_cumprod_prev` | `[1.0, cumprod[:-1]]` | 上一时刻的 ($\bar{\alpha}_{t-1}$)         |
| `sqrt_recip_alphas`   | `sqrt(1/alphas)`      | ($1/\sqrt{\alpha_t}$)，用于去噪时的均值预测     |
| `posterior_variance`  | 公式见 DDPM 论文           | $q(x_{t-1}$)                         |
这些系数会被传入 `sample_ddpm` 函数执行反向扩散采样。


### 条件编码准备：类别 one-hot 模板
```python
onehot = torch.zeros(10, 10, device=device)   # 10x10 单位矩阵
indices = torch.arange(10, ...).view(10, 1)
onehot.scatter_(1, indices, 1)                # 对角线置1

print(f"Total test samples: {len(dataset_test)}")

for batch_idx, data in enumerate(test_loader):
	labels = data["label"].to(device)  # (B,)
	az = data["az"].to(device)  # degrees, (B,)
	names = data["name"]  # list of strings
	
	B = labels.shape[0]

	# class one-hot
	class_onehot = onehot[labels]  # (B,10)
```
预计算一个 10×10 的单位矩阵作为查找表，后续通过 `onehot[labels]` 快速取出每张图对应类别的 one-hot 向量 `(B, 10)`。


### 条件编码准备：方位角傅里叶编码
```python
# azimuth encoding (same as training)

real_angle = torch.deg2rad(az)
real_az_vec = torch.zeros(B, 10, device=device)
for i in range(B):
	real_az_vec[i, 0] = torch.cos(real_angle[i])
	real_az_vec[i, 1] = torch.sin(real_angle[i])
	real_az_vec[i, 2] = torch.cos(2 * real_angle[i])
	real_az_vec[i, 3] = torch.sin(2 * real_angle[i])
	real_az_vec[i, 4] = torch.cos(3 * real_angle[i])
	real_az_vec[i, 5] = torch.sin(3 * real_angle[i])
	real_az_vec[i, 6] = torch.cos(4 * real_angle[i])
	real_az_vec[i, 7] = torch.sin(4 * real_angle[i])
	real_az_vec[i, 8] = torch.cos(5 * real_angle[i])
	real_az_vec[i, 9] = torch.sin(5 * real_angle[i])
```
这是傅里叶特征编码（Fourier feature encoding），将标量的方位角 `az`（度数）映射为一个 10 维向量：`[cos(θ), sin(θ), cos(2θ), sin(2θ), ..., cos(5θ), sin(5θ)]`。

作用：帮助网络学习到方位角的周期性——0° 和 360° 在几何上是同一个角度，这种编码天然保证了 `f(0°) = f(360°)`。


### DDPM 反向采样
```python
samples = sample_ddpm(
    model, betas, alphas_cumprod, alphas_cumprod_prev,
    sqrt_recip_alphas, posterior_variance,
    class_onehot, real_az_vec,        # 条件：类别 + 方位角
    shape=(B, 1, 128, 128),           # 输出形状 (B, C, H, W)
    device=device,
)  # 输出范围 [-1, 1]

samples = (samples + 1.0) / 2.0       # 映射到 [0, 1]
```

`sample_ddpm` 是核心的**反向扩散过程**：
1. 从纯噪声 `x_T ~ N(0, I)` 开始，形状为 `(B, 1, 128, 128)`
2. 从 `t=T` 到 `t=1` 逐步去噪，每一步调用 `model(x_t, t, class_onehot, az_vec)` 预测噪声
3. 用预测的噪声和扩散系数计算 `x_{t-1}`
4. 最终输出 `x_0`，值域 `[-1, 1]`
5. 通过 `(x+1)/2` 映射到 `[0, 1]`，方便 `save_image` 保存


### 保存图像
```python
# save images: save_dir/label/name_az.png (like old generate)
for i in range(B):
    label_i = int(labels[i].item())
    name_i = names[i]
    az_i = float(az[i].item())

    class_dir = os.path.join(args.save_dir, str(label_i))
    os.makedirs(class_dir, exist_ok=True)

    filename = f"{name_i}_{az_i:.2f}.png"
    save_path = os.path.join(class_dir, filename)
    save_image(samples[i], save_path)

torch.cuda.empty_cache()  # 每批释放一次显存，防止长时间运行时 OOM
```

```
SAMPLE/generated/
├── 0/
│   ├── sample_001_45.00.png
│   └── sample_002_90.00.png
├── 1/
│   └── sample_003_30.00.png
├── ...
└── 9/
```
- 按**类别**分子文件夹（0-9）
- 文件名包含**原始样本名 + 方位角**，便于追溯生成条件