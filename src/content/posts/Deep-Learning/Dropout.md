---
title: "Dropout"
description: "Dropout 的原理和 PyTorch 实现"
date: 2026-09-19
tags: ["深度学习", "Dropout"]
draft: false
---

## ☕️引言

Dropout 目的是防止模型**过拟合**（Overfitting），做法就是**在训练时**按照一定的概率 $p$ 将某些神经元丢弃，使得后面的神经元不依赖前面的某些神经元，因此能有效防止模型过拟合，分为两个阶段：

- **训练阶段**：以概率 p 丢弃某些神经元，然后除以 `(1 - p)`，保持输出的期望值不变；
- **推理阶段**：等值映射，即输入 x，就输出 x。

![Dropout](/tech-blog/images/Deep-Learning/Dropout/Dropout.png)

📒训练时采样伯努利分布：

$$
m \sim \text{Bernoulli}(p)
$$

当 $m=1$ 时丢弃神经元，当 $m=0$ 时保留神经元，因此输出的期望值为：

$$
E[mx]=xE[m]=x\times(1-p)
$$

为了保持输出期望值不变，乘以一个缩放因子 `1 / (1 - p)`，放大激活值。

这种实现方式称为 Inverted Dropout，是当前 PyTorch 的标准实现方式，跟**原始Dropout 操作**（推理时✖️p）是等价的。

![Dropout](/tech-blog/images/Deep-Learning/Dropout/Dropout-train-eval.png)

## 🧑‍💻Implementation

下面是 Dropout 的 PyTorch 实现代码：

```python
import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super(Dropout, self).__init__()
        self.p = p

    def forward(self, x: torch.Tensor):
        if not self.training or self.p == 0:
            return x

        # Bernoulli mask & scaling
        out = torch.where(torch.rand_like(x) > self.p, x, 0)
        return out / (1 - self.p)
```

## Reference

[1] Nitish Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overﬁtting. University of Toronto. 2014.