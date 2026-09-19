---
title: "归一化层（Normalization）"
description: "不同归一化层的原理、区别和实现"
date: 2026-09-19
tags: ["深度学习", "Normalization"]
draft: false
---

## ☕️引言

归一化层（Normalization）对于模型训练的稳定性、效率有很重要的作用，这里介绍常见的 3 种归一化方法：**批量归一化**（BatchNorm）、**层归一化**（LayerNorm）、**均方根层归一化**（RMSNorm）。

## BatchNorm

BatchNorm 的目的是解决 **内部协变量偏移**（Internal Covariate Shift），即模型每一层的输入分布在训练时不断变化，通过对网络内部的激活值进行标准化，使深层网络**更容易**、**更快地**优化，BatchNorm 常用于卷积神经网络（CNNs）。

BatchNorm 的具体步骤如下：

1. 计算批量数据的均值和方差；
2. 归一化；
3. 缩放和偏移操作（📒**重要**‼️：用于恢复模型原有的表示能力）。

![BatchNorm](/tech-blog/images/Deep-Learning/BatchNorm/BatchNorm.png)

BatchNorm 在训练时和推理时的逻辑是不一致的，在训练时均值和方差统计量在 mini-batch 中估计，同时维护一个全局统计量 `running_mean` 和 `running_var`，在推理时使用 `running_mean` 和 `running_var` 作为 BatchNorm 的均值和方差。

此外，对于 2D 的情况，即输入形状为 `(B, C, H, W)`，Batch 的维度实际为 `C`（通道维度）。

🧑‍💻下面是 `BatchNorm1D` 的具体实现，输入形状为：`(B, D)`，📒注意 `running_var` 的更新使用**无偏估计**的 `var`

```python
class BatchNorm1D(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super(BatchNorm1D, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # Input: [B, D]
        # 1. Calculate average and stdandard deviation.
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            with torch.no_grad():
                B = x.size(0)
                unbiased_var = var * B / (B - 1)
                # update running_mean and running_var
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var)
        else:
            mean = self.running_mean
            var = self.running_var

        # 2. Normalize
        z = (x - mean) / torch.sqrt(var + self.eps)
        # 3. Scale and shift
        return self.gamma * z + self.beta
```

下面是 `BatchNorm2D` 的实现：

```python
class BatchNorm2D(nn.Module):
    def __init__(self, dim: int, momentum: float = 0.1, eps: float = 1e-5):
        super(BatchNorm2D, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: torch.Tensor):
        # Input: [B, C, H, W]
        # 1. Calculate average and stdandard deviation.
        if self.training:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            with torch.no_grad():
                N = x.size(0) * x.size(2) * x.size(3)
                unbiased_var = var * N / (N - 1)
                # update running_mean and running_var
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * unbiased_var)
        else:
            mean = self.running_mean
            var = self.running_var
        
        mean = mean.view(1, -1, 1, 1)
        var = var.view(1, -1, 1, 1)
        # 2. Normalize
        z = (x - mean) / torch.sqrt(var + self.eps)
        # 3. Scale and shift
        return self.gamma * z + self.beta
```

## LayerNorm

虽然 BatchNorm 能让模型训练更稳定、更迅速，但是有如下局限性：

- 需要记录均值和方差统计量的**滑动平均**；
- 性能对批量大小（batch size）敏感度较高（📒batch size 过小会导致均值方差统计量计算不准确）；
- 难以适配到 RNNs、Transformer 等序列长度可变的场景。

LayerNorm 的具体步骤如下：

1. 计算单个 sample 均值和方差；
2. 归一化；
3. 缩放和偏移操作（📒**重要**‼️：用于恢复模型原有的表示能力）。

📒**注意**：LayerNorm 的训练和推理阶段的逻辑是一致的，广泛应用于现代 Transformer 模型。

🧑‍💻LayerNorm 的具体实现如下（相对于 BatchNorm 更简单，**训练和推理模型逻辑一致**）：

```python
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # Input: [B, L, D]
        # 1. Calculate average and stdandard deviation.
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        # 2. Normalize
        z = (x - mean) / (var + self.eps)
        # 3. Scale and shift
        return self.gamma * z + self.beta
```

## RMSNorm

RMSNorm 可以视为 LayerNorm 的简化版，消除了 LayerNorm 的去中心化操作，不需要计算 sample 的均值，优势如下：

- 参数量更小，只需要缩放参数 `gamma`；
- 计算效率高，不需要计算样本均值；
- RMSNorm 对于输入 x 具有**尺度不变性**，即 `RMSNorm(x) = RMSNorm(ax)`，其中 a 为一个常数。

📒其中尺度不变性是因为均方根的性质：

$$
\text{RMS}(\alpha x)=\alpha\text{RMS}(x)
$$

而 RMSNorm 的计算公式如下：

$$
\text{RMSNorm}(x)=\frac{x}{\text{RMS}(x)}\odot \gamma
$$

因此

$$
\begin{aligned}
\text{RMSNorm}(\alpha x)&=\frac{\alpha x}{\text{RMS}(\alpha x)}\odot \gamma\\
&=\frac{\alpha x}{\alpha\text{RMS}(x)}\odot \gamma\\
&=\frac{x}{\text{RMS}(x)}\odot \gamma\\
&=\text{RMSNorm}(x)
\end{aligned}
$$

🧑‍💻RMSNorm 的 pytorch 实现代码如下（非常简洁👏）：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # Input: [B, L, D]
        var = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + self.eps)
        out = x * inv_rms * self.gamma
        return out
```

## References

[1] Sergey Ioffe et al. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. Google. 2015.

[2] Jimmy Lei Ba et al. Layer Normalization. University of Toronto. 2016.

[3] Biao Zhang et al. Root Mean Square Layer Normalization. University of Edinburgh. 2019.