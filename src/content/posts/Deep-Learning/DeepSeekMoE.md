---
title: "DeepSeekMoE"
description: "DeepSeek 混合专家模型原理和实现"
date: 2026-09-13
tags: ["深度学习"]
draft: false
---

## ☕️引言

**Mixture-of-Experts** (MoE)，即混合专家模型，通常是指在 Transformer 里面的 MLP 层采用多个**并列的分支**，每个 token 经过 MoE 层后的输出是多个分支计算结果的**加权和**。DeepSeekMoE 相比于之前的 MoE 模型（如 GShard），在相同的参数下，采用 **更多的路由专家** 以及 **共享专家**（每次都会激活），目的是让每个专家**更加专精** 且**降低专家冗余度**！

![DeepSeekMoE](/tech-blog/images/Deep-Learning/DeepSeekMoE/DeepSeekMoE.png)

## 负载均衡

在 MoE 模型训练时通常需要添加**负载均衡**损失，可以提升每个专家训练的概率，避免少量专家得到大部分的训练，具有如下形式：
$$
\mathcal{L}_\text{balance}=\sum_{i=1}^N f_i\cdot p_i
$$
其中 $f_i$ 为选择专家 $i$ 的相对频率，具体表达式如下：
$$
f_i=\frac{n_i}{\sum_{k=1}^N n_k}
$$
而 $p_i$ 则是每个 token 分配给专家 $i$ 的平均概率（topk 选择之前），下面是对应的 PyTorch 实现代码：

```python
def deepseek_moe_loss(
    logits: torch.Tensor,  # [B, T, E]
    top_k: int,
    alpha: float = 1e-4,
):
    _, seq_len, num_experts = logits.shape
    scores = logits.float().sigmoid()
    probs = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p = probs.mean(dim=1)

    with torch.no_grad():
        indices = scores.topk(top_k, dim=-1).indices
        counts = F.one_hot(indices, num_classes=num_experts).sum(dim=(1, 2))
        f = counts.float() * (num_experts / (top_k * seq_len))

    return alpha * (f * p).sum(dim=-1).mean()
```

## 🧑‍💻代码实现

DeepSeekMoE 的 **具体 PyTorch 简化实现如下**：

```python
import torch
import torch.nn as nn
from torch.nn import functional as F


class DeepSeekMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super(DeepSeekMLP, self).__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.swish = nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self.down_proj(self.swish(self.gate_proj(x)) * self.up_proj(x))


class DeepSeekMoERouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super(DeepSeekMoERouter, self).__init__()
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.register_buffer("correction_bias", torch.zeros(num_experts))

    def forward(self, x: torch.Tensor):
        logits = self.router(x)
        scores = F.sigmoid(logits)
        scores_for_choice = scores + self.correction_bias
        topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )[1]
        topk_weights = torch.gather(scores, 1, topk_indices)
        if self.norm_topk_prob:
            topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-12

        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_weights, topk_indices


class DeepSeekMoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        num_shared_experts: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
    ):
        super(DeepSeekMoELayer, self).__init__()
        self.experts = nn.ModuleList(
            [DeepSeekMLP(hidden_size, intermediate_size) for _ in range(num_experts)]
        )
        self.router = DeepSeekMoERouter(
            hidden_size, num_experts, top_k, norm_topk_prob, routed_scaling_factor
        )
        self.shared_experts = DeepSeekMLP(
            hidden_size, intermediate_size * num_shared_experts
        )
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_size = x.size()
        x = x.view(-1, hidden_size)
        # 1. pass through shared experts
        shared_output = self.shared_experts(x)

        # 2. pass through routed experts
        routed_output = torch.zeros_like(x)
        topk_weights, topk_indices = self.router(x)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(topk_indices == expert_id)
            if token_ids.numel() == 0:
                continue

            expert_input = x[token_ids]
            expert_output = expert(expert_input)

            expert_weights = topk_weights[token_ids, slots].unsqueeze(-1)
            weighted_output = expert_weights * expert_output

            routed_output.index_add_(
                0, token_ids, weighted_output.to(routed_output.dtype)
            )
        out = shared_output + routed_output
        out = out.reshape(batch_size, seq_len, hidden_size)
        return out
```

## References

[1] DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. DeepSeek AI. 2024.