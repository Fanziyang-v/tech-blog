---
title: "直接偏好优化（DPO）"
description: "📝动手推导 DPO 公式"
date: 2026-09-13
tags: ["强化学习", "DPO"]
draft: false
---

## ☕️引言

**直接偏好优化**（Direct Preference Optimization, DPO） 是一种稳定、高效的强化学习算法。

传统 RLHF 是一种较为复杂且不稳定的过程——首先训练一个符合人类偏好的**奖励模型**（Reward Model），然后通过强化学习微调一个语言模型，使其奖励最大化并与原始模型偏离程度较小。在数学上，DPO 与 RLHF 优化的目标是一致的，但是 DPO 更加简单，因为不需要训练奖励模型。

**DPO 大致流程：**

1. 对于每一个 prompt $x$，采样 $y_1,y_2\sim \pi_{\text{ref}}(\cdot|x)$，构造**偏好数据集** $\mathcal{D}=\lbrace (x^{(i)},y_w^{(i)},y_l^{(i)}) \rbrace_{i=1}^N$
2. 给定引用模型 $\pi_{\text{ref}}$、偏好数据集 $\mathcal{D}$、正则化系数 $\beta$，优化语言模型 $\pi_\theta$，最小化 DPO 损失 $\mathcal{L}_{\text{DPO}}$
   1. 若  $\pi_{\text{ref}}$ 可用，则将语言模型 $\pi_\theta$ 初始化为 $\pi_{\text{ref}}$ ；
   2. 否则，通过最大化偏好 $(x, y_w)$ 似然（ $\pi_{\text{ref}}=\arg\max_\pi \mathbb{E}_{x,y_w\sim \mathcal{D}}[\log \pi(y_w|x)]$ ）

![DPO-vs-RLHF](/tech-blog/images/Reinforcement-Learning/DPO/DPO-vs-RLHF.png)

## ⭐️公式推导

**DPO 的损失函数为：**

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_\text{ref})=-\mathbb{E}_{(x,y_w,y_l)\sim \mathcal{D}}[\log \sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)})]
$$

### RLHF 流程

RLHF 通常包含三个阶段：（1）有监督微调（SFT）；（2）奖励学习；（3）强化学习。以 Bradley-Terry (BT) 偏好模型为例，生成符合人类偏好的输出概率分布为：

$$
p^*(y_1\succ y_2|x)=\frac{\exp{r^*(x,y_1)}}{\exp{r^*(x,y_1)}+\exp{r^*(x,y_2)}}
$$

给定一个从 $p^*$ 采样的偏好数据集 $\mathcal{D}=\lbrace (x^{(i)},y_w^{(i)},y_l^{(i)}) \rbrace_{i=1}^N$，通过极大似然估计优化一个奖励模型 $r_\phi(x,y)$，本质上就是一个**二元交叉熵**（Binary Cross Entropy）损失函数：

$$
\mathcal{L}_R(r*\phi,\mathcal{D})=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}[\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]
$$

具体推导过程如下：


$$
\begin{aligned}
\mathcal{L}_R(r_\phi,\mathcal{D})&=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\log p^*(y_1\succ y_2|x)\\
&=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\log \frac{\exp{r^*(x,y_1)}}{\exp{r^*(x,y_1)}+\exp{r^*(x,y_2)}}\\
&=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}[\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]
\end{aligned}
$$

其中 $\sigma(x)=1/(1+e^{-x})$ 是 sigmoid 函数。

**强化学习阶段**：在强化学习阶段，优化的目标如下（最大化奖励 + KL 散度限制）：
$$
\max_{\pi_\theta} \underbrace{\mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r_\phi(x,y)]}_{\text{Reward term}}-\underbrace{\beta\cdot\mathbb{D}_{\text{KL}}[\pi_\theta(y|x) \Vert\pi_{\text{ref}}(y|x)]}_{\text{Regularization term}}
$$

其中 $\beta>0$ 是正则化强度系数，防止策略模型与引用模型偏离过大。

### 直接偏好优化

从强化学习的目标开始进行推导：

$$
\begin{aligned}
 &\max_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r_\phi(x,y)]-\beta\mathbb{D}_{\text{KL}}[\pi_\theta(y|x) \Vert\pi_{\text{ref}}(y|x)]\\
 =&\max_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}]\\
 =&\min_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}-\log{\exp(\frac{1}{\beta}r_\phi(x,y))}]\\
 =&\min_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)\exp(\frac{1}{\beta}r_\phi(x,y))\frac{1}{Z(x)}Z(x)}]\\
 =&\min_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D},y\sim \pi_\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi^*(y|x)Z(x)}]\\
 =&\min_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D},y\sim \pi^*\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi^*(y|x)}-\log Z(x)]\\
 =&\min_{\pi_\theta} \mathbb{E}_{x\sim \mathcal{D},y\sim \pi^*\theta(y|x)}[\log\frac{\pi_\theta(y|x)}{\pi^*(y|x)}]\\
 =&\mathbb{D}_\text{KL}(\pi_\theta(y|x)\Vert \pi^*(y|x))
\end{aligned}
$$

其中函数 $Z(x)$ 与策略模型 $\pi_\theta$ 无关，因此可以直接消去，其表达式为：

$$
 Z(x)=\sum_y \pi_\text{ref}(y|x)\exp{(\frac{1}{\beta}r(x,y))}
$$

$$
 \pi^*(y|x)=\pi_\text{ref}(y|x)\exp(\frac{1}{\beta}r_\phi(x,y))\frac{1}{Z(x)}
$$

**理论的最优解是**：
$$
 \pi_\theta(y|x)=\pi^*(y|x)=\pi_\text{ref}(y|x)\exp(\frac{1}{\beta}r_\phi(x,y))\frac{1}{Z(x)}
$$

进行恒等变化，可以得到**奖励函数的表达式**：

$$
 r_\phi(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}+\beta\log Z(x)
$$

将其代入到偏好学习损失函数中，可以得到：

$$
\begin{aligned}\mathcal{L}_\text{DPO}(\pi_\theta;\pi_\text{ref})&=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}[\log \sigma(r_\phi(x,y_w)-r_\phi(x,y_l))]\\
&=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}[\log \sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)})]\\
&=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}[\log \sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}-\beta\log\frac{\pi_\text{ref}(y_w|x)}{\pi_\text{ref}(y_l|x)})]
\end{aligned}
$$

## 🧑‍💻PyTorch 实现代码

下面给出在 LLM 场景下的 DPO 损失函数的 PyTorch 实现代码，先计算序列级别的

### DPO Loss

**输入**：策略模型和参考模型的输出对数概率

**输出**：DPO 损失

**计算步骤**：

1. 计算策略模型和参考模型的对数概率比例 `logratios`；
2. 计算 `logits = beta * (policy_log_ratios - ref_log_ratios)`
3. 计算 DPO 损失：`losses = -F.logsigmoid(logits)`
4. 平均 batch 维度：`losses.mean()`

```python
import torch
from torch.nn import functional as F

def dpo_loss(
    policy_chosen_log_probs: torch.Tensor,
    policy_rejected_log_probs: torch.Tensor,
    ref_chosen_log_probs: torch.Tensor,
    ref_rejected_log_probs: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    # Input tensor shape: [B, L-1]
    policy_log_ratios = policy_chosen_log_probs - policy_rejected_log_probs
    ref_log_ratios = ref_chosen_log_probs - ref_rejected_log_probs
    logits = beta * (policy_log_ratios - ref_log_ratios)
    loss = -F.logsigmoid(logits).mean()
    return loss
```



### Sequence-level log probs

通常对于 (M)LLMs 来说，输出都是一个序列，因此需要有一个 sequence-level 的 DPO loss，实际上就是将 token-level 的 log probs 先聚合（求和或平均），注意 `labels=-100`（padding / prompt 位置）不需要计算 log probs（这些位置不参与梯度计算）。

```python
def sequence_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~mask, 0).sum(-1)

def sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # logits: model output, of shape [B, L]
    # labels: token ids, of shape [B, L]
    # mask: True if token used in DPO loss, i.e., response tokens.
    #       False if masked, i.e, prompt and padding tokens.
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    shifted_mask = mask[:, 1:].bool()
    safe_labels = shifted_labels.masked_fill(~shifted_mask, 0)

    token_log_probs = F.log_softmax(shifted_logits, dim=-1, dtype=torch.float32)
    token_log_probs = token_log_probs.gather(
        -1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return sequence_sum(token_log_probs, mask)
```

### Main DPO Loss

有了 `dpo_loss` 和 `sequence_log_probs` 两个接口之后，实现 `main_dpo_loss` ，接受策略模型和参考模型的输出 `logits` 以及 `labels` 作为输入，然后先获取序列级别的对数概率，再调用 `dpo_loss` 计算 DPO 损失。

```python
def main_dpo_loss(
    policy_chosen_logits: torch.Tensor,
    policy_rejected_logits: torch.Tensor,
    ref_chosen_logits: torch.Tensor,
    ref_rejected_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    chosen_mask: torch.Tensor,
    rejected_mask: torch.Tensor,
    beta: float = 0.1,
):
    # 1. Get sequence-level log-probabilities
    policy_chosen_log_probs = sequence_log_probs(
        logits=policy_chosen_logits, labels=chosen_labels, mask=chosen_mask
    )
    policy_rejected_log_probs = sequence_log_probs(
        logits=policy_rejected_logits, labels=rejected_labels, mask=rejected_mask
    )
    ref_chosen_log_probs = sequence_log_probs(
        logits=ref_chosen_logits, labels=chosen_labels, mask=chosen_mask
    )
    ref_rejected_log_probs = sequence_log_probs(
        logits=ref_rejected_logits, labels=rejected_labels, mask=rejected_mask
    )
    # 2. Compute DPO loss
    loss = dpo_loss(
        policy_chosen_log_probs=policy_chosen_log_probs,
        policy_rejected_log_probs=policy_rejected_log_probs,
        ref_chosen_log_probs=ref_chosen_log_probs,
        ref_rejected_log_probs=ref_rejected_log_probs,
        beta=beta,
    )

    return loss
```

## 📜参考文献

[1] Rafael Rafailov et.al. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. 2024.

[2] RethinkFun. [DPO (Direct Preference Optimization) 算法讲解](https://www.bilibili.com/video/BV1GF4m1L7Nt/?spm_id_from=333.337.search-card.all.click&vd_source=c8a32a5a667964d5f1068d38d6182813). 2024.

