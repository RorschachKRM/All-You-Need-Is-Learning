---
title: 训练过程
tags:
  - DDPM
  - Training
related:
  - "[[DDPM - General Framework#7. 模块五：训练管线（Training Pipeline）]]"
---

# 代码1
（代码 1 来源：EM_deeplearning_beifen\DDPM\train_ddpm.py)
```python
import sample_dataset
from ddpm_model import (
    ConditionalUNet,
    make_beta_schedule,
    extract,
    sample_ddpm,
)
```


## 参数解析
```python
def parameter_setting(args):
    config = {}
    config["bs"] = args.bs              # batch size, 默认 64
    config["lr"] = args.lr              # learning rate, 默认 1e-4
    config["num_epochs"] = args.num_epochs  # 训练轮数, 默认 5000
    config["save_dir"] = args.save_dir   # 保存目录, 默认 "SAMPLE/"
    config["timesteps"] = args.timesteps  # 扩散步数, 默认 1000
    config["beta_schedule"] = args.beta_schedule  # 调度策略, 默认 "linear"

    return config

```


## build_diffusion：构建扩散系数
```python
def build_diffusion(timesteps: int, device: torch.device):
	# 计算所有需要的系数:
    betas = make_beta_schedule(timesteps).to(device)  # β_t, (T,)
    alphas = 1.0 - betas   # α_t = 1-β_t, (T,)
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # ᾱ_t = ∏α_s, (T,)
    alphas_cumprod_prev = torch.cat(
        [torch.tensor([1.0], device=device), alphas_cumprod[:-1]], dim=0
    )     # ᾱ_{t-1}, (T,)

	# 关键系数:
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # √ᾱ_t
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod) # √(1-ᾱ_t)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)   # 1/√α_t

    posterior_variance = (
        betas
        * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod + 1e-8)
    )   # σ_t² = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)

	# 返回字典包含所有系数，供训练和采样使用:
    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }
```

**系数用途速查**：

| 系数                              | 用途                                                                            |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `sqrt_alphas_cumprod`           | 前向加噪中的原图系数：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ |
| `sqrt_one_minus_alphas_cumprod` | 前向加噪中的噪声系数                                                                    |
| `sqrt_recip_alphas`             | 反向均值计算：$1/\sqrt{\alpha_t}$                                                    |
| `posterior_variance`            | 反向采样时的方差                                                                      |


## train_ddpm — 训练主循环
```python
def train_ddpm(config):
    batch_size = config["bs"]
    lr = config["lr"]
    train_epoch = config["num_epochs"]
    timesteps = config["timesteps"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data_loader
    data_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset_train = sample_dataset.SAMPLE_Dataset(
        txt_file="data/sample_train_90.txt",
        transform=data_transforms,
    )
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # model
    model = ConditionalUNet(img_channels=1, base_ch=64, time_dim=256, cond_dim=64).to(
        device
    )

    diffusion = build_diffusion(timesteps, device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    # results save folder
    root = config["save_dir"]
    model_name = "SAMPLE_DDPM_"
    if not os.path.isdir(root):
        os.mkdir(root)
    if not os.path.isdir(os.path.join(root, "generated_results")):
        os.mkdir(os.path.join(root, "generated_results"))

    writer = SummaryWriter("./DDPM")

    # label one-hot template
    onehot = torch.zeros(10, 10)
    onehot = onehot.scatter_(
        1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10, 1), 1
    )

    print("DDPM training start!")
    start_time = time.time()

    betas = diffusion["betas"]

    alphas_cumprod = diffusion["alphas_cumprod"]

    sqrt_alphas_cumprod = diffusion["sqrt_alphas_cumprod"]

    sqrt_one_minus_alphas_cumprod = diffusion["sqrt_one_minus_alphas_cumprod"]

  

    for epoch in range(train_epoch):

        epoch_start_time = time.time()

        losses = []

  

        for idx, data in enumerate(train_loader):

            model.train()

            optimizer.zero_grad()

  

            x0 = data["image"].to(device)  # (B,1,H,W), in [0,1]

            labels = data["label"]  # CPU tensor

            az = data["az"].to(device)  # degrees, (B,)

  

            # scale images to [-1, 1]

            x0 = x0 * 2.0 - 1.0

  

            # class one-hot

            class_onehot = onehot[labels].to(device)  # (B,10)

  

            # azimuth encoding (10-dim cos/sin up to 5th harmonic)

            real_angle = torch.deg2rad(az)

            B = x0.shape[0]

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

  

            # sample t uniformly for each batch element

            t = torch.randint(0, timesteps, (B,), device=device).long()

  

            noise = torch.randn_like(x0)

            sqrt_alphacum_t = extract(sqrt_alphas_cumprod, t, x0.shape)

            sqrt_one_minus_alphacum_t = extract(

                sqrt_one_minus_alphas_cumprod, t, x0.shape

            )

            xt = sqrt_alphacum_t * x0 + sqrt_one_minus_alphacum_t * noise

  

            noise_pred = model(xt, t, class_onehot, real_az_vec)

            loss = mse_loss(noise_pred, noise)

  

            loss.backward()

            optimizer.step()

  

            losses.append(loss.item())

  

        epoch_end_time = time.time()

        per_epoch_ptime = epoch_end_time - epoch_start_time

  

        mean_loss = sum(losses) / max(len(losses), 1)

        print(

            "[Epoch %d/%d] - ptime: %.2f, loss: %.6f"

            % (epoch + 1, train_epoch, per_epoch_ptime, mean_loss)

        )

        writer.add_scalar("DDPM_loss/train", mean_loss, epoch + 1)

  

        # 每 10 轮生成一次样本

        if (epoch + 1) % 10 == 0 or epoch == train_epoch - 1:

            save_path = os.path.join(

                root,

                "generated_results",

                model_name + str(epoch + 1) + ".png",

            )

            # generate 16 samples and save as a 4x4 image grid

            samples = generate_grid_samples(

                model,

                diffusion,

                device,

            )  # (16,1,128,128), in [-1,1]

            samples = (samples + 1.0) / 2.0  # 映射到 [0,1] 方便保存

            grid = make_grid(samples, nrow=4, normalize=False)

            save_image(grid, save_path)

  

    print("DDPM Training finish!... save model")

    torch.save(model.state_dict(), os.path.join(root, model_name + "ddpm_unet_param.pkl"))
```