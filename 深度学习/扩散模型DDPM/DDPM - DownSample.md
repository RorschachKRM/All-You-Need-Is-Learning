---
title: 下采样
tags:
  - DDPM
  - DownSample
  - Convolution
related:
  - "[[CNN知识点]]"
---


#  1. 步长卷积实现
 此处是使用步长卷积（步长为 2 ，padding=same的 3×3 卷积）实现下采样：
 （代码来源：\DDPM\ddpm_model.py）
```python
class DownSample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
```
