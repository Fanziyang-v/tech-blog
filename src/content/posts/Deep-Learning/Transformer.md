---
title: "Transformer"
description: "🧑‍💻动手实现 Transformer"
date: 2026-09-13
tags: ["深度学习"]
draft: false
---

## ☕️引言

Transformer 模型架构最初由 Google 于 2017 年的一篇论文 《Attention Is All You Need》提出，逐渐成为深度学习各个领域的标准模型架构，例如，Qwen、Gemini、GPT、Claude 等现代大模型都基于 Transformer 架构，本文将介绍 Transformer 模型的核心组件，并基于 PyTorch 给出具体实现代码。

**📒详细代码可见**：https://github.com/Aporifold/Transformer

## 🧠模型架构

Transformer 结构如下图所示，Transformer 遵循编码器-解码器（Encoder-Decoder）的结构，每个 Transformer Block 的结构基本上相同，其编码器和解码器可以视为两个独立的模型，例如：ViT 仅使用了 Transformer 编码器，而 GPT  仅使用了 Transformer 解码器。 

![Transformer](/tech-blog/images/Deep-Learning/Transformer/Transformer.png)

### 编码器

编码器包含 $N=6$ 相同的层，每个层包含两个子层，分别是多头自注意力层（Multi-Head Self-Attention）和前馈神经网络层（Feed-Forward Network），每个子层都包含残差连接（Residual Connection）和层归一化（Layer Normalization），使模型更容易学习。FFN 层是一个两层的多层感知机（Multilayer Perceptron）。
### 解码器

解码器也包含 $N=6$ 相同的层，包含三个子层，分别是掩码多头自注意力层（Masked Multi-Head Attention）、编码器-解码器多头注意力层（Cross Attention）和前馈神经网络 层。

其中，掩码多头自注意力层用于将输出的 token 进行编码，在应用注意力机制时使用**因果掩码**，以保持模型的自回归（Auto Regressive）特性，即前面的 token 不能注意到后面的 token，编码后作为 Cross Attention 层的 **Query**，而 Cross Attention 层的 Key 和 Value 来自于编码器的输出，最后通过 FFN 层产生解码器块的输出。


### 位置编码

与 RNNs 串行处理序列信息的方式不同，**注意力机制本身不包含位置关系**，因此 Transformer 需要为序列中的每个 token 添加位置信息，Transformer 中使用了**正余弦位置编码**（Sinusoidal Position Embedding ），位置编码由以下数学表达式给出：

$$
\begin{aligned}
PE_{pos,2i}&=\sin(\frac{pos}{10000^{2i/d_\text{model}}})\\
PE_{pos,2i+1}&=\cos(\frac{pos}{10000^{2i/d_\text{model}}})
\end{aligned}
$$

其中，`pos` 为 token 所在的序列位置，`i` 则是对应的特征维度。

其 Pytorch 实现代码如下：

```python
class PositionalEncoding(nn.Module):
    """Transformer Sinusoidal Position Embeddings."""

    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._init_position_embeddings()

    def _init_position_embeddings(self):
        _2i = torch.arange(0, self.d_model, step=2)
        pos = torch.arange(self.max_len)
        inv_freqs = 1 / 10000 ** (_2i / self.d_model)
        args = pos.unsqueeze(1) * inv_freqs.unsqueeze(0)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        pos_emb = self.pe[:seq_len].unsqueeze(0)
        return pos_emb
```

### ⭐️注意力机制

注意力机制出现在 Transformer 之前，包括两种类型：加性注意力和乘性注意力。

Transformer 使用的是常见的乘性注意力，首先计算点积相似度，然后通过 Softmax 后得到注意力权重，根据注意力权重对 Values 进行加权求和，具体的过程可以表示为以下数学公式：

$$
\text{Attention}(Q,K,V)=\text{Softmax}(\frac{QK^T}{\sqrt{d}})V
$$

![Transformer](/tech-blog/images/Deep-Learning/Transformer/Attention.png)

其中 $Q,K,V\in\mathbb{R}^{n\times d}$，注意力计算时还可以添加一个可选的 Attention Mask（例如 padding、causal mask），防止 token 注意到不应该注意到的 tokens。

注意力计算中包含了一个温度参数 $d$ ，**避免点积的结果过大或过小**，导致 softmax 后的结果梯度几乎为 0 的区域，降低模型的收敛速度，Scaled Dot-Product Attention 的 PyTorch 实现代码如下：

```python
class ScaleDotProductAttention(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Scaled-Dot Product Attention.

        Args:
            q (torch.Tensor): Query tensor, of shape (B, H, L, D)
            k (torch.Tensor): Key tensor, of shape (B, H, L, D)
            v (torch.Tensor): Value tensor, of shape (B, H, L, D)
            mask (torch.Tensor, optional): Boolean attention mask, 
                of shape (B, 1, 1, L). Defaults to None.

        Returns:
            tuple (torch.Tensor, torch.Tensor): A tuple of attention outputs and attention weights
        """
        d_k = k.size(-1)
        scores: torch.Tensor = q @ k.transpose(-1, -2) * d_k**-0.5

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        out = attn_weights @ v
        return out, attn_weights
```

此外，Transformer 使用**多头注意力机制**，捕捉序列的不同模式，类似于卷积层（有多个卷积核），可表示为：

$$
\begin{aligned}
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\text{head}_2,\dots,\text{head}_h)W^O\\
\text{where }\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\end{aligned}
$$

多头注意力机制的 PyTorch 实现如下：

```python
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Module."""

    def __init__(self, d_model: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.attn = ScaleDotProductAttention()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = self.d_model // self.n_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask=None):
        # 1. QKV projection and split into H heads.
        q = self._split(self.q_proj(q))
        k = self._split(self.k_proj(k))
        v = self._split(self.v_proj(v))

        # 2. Apply scaled-dot product attention
        out, _ = self.attn(q, k, v, mask=mask)

        # 3. Concat each attention heads.
        out = self._concat(out)
        return out

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # split tensor into H heads.
        batch_size, seq_len, hidden_dim = x.size()
        out = x.view(batch_size, seq_len, self.n_heads, self.d_head)
        return out.transpose(-2, -3)

    def _concat(self, x: torch.Tensor) -> torch.Tensor:
        # batch_size, n_heads, seq_len, d_head
        batch_size, n_heads, seq_len, d_head = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
```

