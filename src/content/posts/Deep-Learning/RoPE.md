---
title: "旋转位置编码"
description: "深入理解旋转位置编码（RoPE）的原理"
date: 2026-09-13
tags: ["深度学习", "位置编码"]
draft: false
---

## ☕️引言

旋转位置编码 (Rotary Position Embedding, RoPE) 是现代主流大模型（如 Qwen、DeepSeek、Kimi 等）的首选位置编码方式，RoPE 最关键的数学性质是：**通过对每个 token 独立施加等范数旋转，让 query 和 key 的内积依赖相对位置**。RoPE 把相对位置建模与点积注意力结合得十分简洁：

$$
R_m^TR_n=R_{n-m}
$$

**Preview**：RoPE 的做法是将 $d$ 维空间切分为 $d/2$ 个 2D 子空间，对每个 2D 子空间的向量进行不同频率的旋转，由此嵌入相对位置信息。

## 📚预备知识

假设 $\mathbb{S}_N=\{ w_i \}_{i=1}^N$ 是一个输入 Token 序列，其中 $w_i$ 为第 $i$ 个元素，对应的 word embeddings 表示为 $\mathbb{E}_N=\{x_i\}_{i=1}^N$，其中 $x_i\in\mathbb{R}^d$ 是 token $w_i$ 的不带位置编码信息的 d 维词嵌入向量，注意力机制首先会注入位置编码信息到 word embeddings 并将其转换为 queries、keys、values：

$$
\begin{aligned}
q_m&=f_q(x_m,m)\\
k_n&=f_k(x_n,n)\\
v_n&=f_v(x_n,n)
\end{aligned}
$$

其中 $q_m$，$k_n$ 和 $v_n$ 通过 $f_q$、$f_k$、$f_v$ **融合位置信息 m 和 n**，然后使用 $q$ 和 $k$ 计算注意力权重，输出为 values 的加权和：

$$
\begin{aligned}
a_{m,n}&=\frac{\exp(\frac{q_m^Tk_n}{\sqrt{d}})}{\sum_{j=1}^N\exp(\frac{q_m^Tk_j}{\sqrt{d}})}\\
o_m&=\sum_{n=1}^Na_{m,n}v_n
\end{aligned}
$$

### 绝对位置编码

原始 Transformer 提出**绝对位置编码**：
$$
f_{t:t\in\{q,k,v\}}(x_i,i):=W_{t:t\in\{q,k,v\}}(x_i+p_i)
$$

其中 $p_i\in \mathbb{R}^d$ 是一个 d 维位置编码，仅跟 token $x_i$ 和维度 d 相关，采用正余弦编码方式：

$$
\begin{cases}
p_{k,2t}&=\sin (\frac{k}{10000^{2t/d}})\\
p_{k,2t+1}&=\cos (\frac{k}{10000^{2t/d}})
\end{cases}
$$

其中 $k$ 是 token 在序列中的位置，$2t$ / $2t+1$ 是在位置编码向量中的位置（偶数位置用 sin、奇数位置用 cos）。

## 旋转位置编码

目标是将 $q$ 和 $k$ 的内积（约等于注意力计算）**仅包含相对位置信息**：

$$
<f_q(x_m,m),f_k(x_n,n)>=g(x_m,x_n,m-n)
$$


![RoPE](/tech-blog/images/Deep-Learning/RoPE/RoPE.png)

### 🤗简单场景 (2D)

考虑 $d=2$ 的简单例子，在极坐标系下，任何 2D 向量都可以用**模长和角度**表示，2D 情况下的解为：
$$
\begin{aligned}
f_q(x_m,m)&=(W_qx_m)e^{im\theta}\\
f_k(x_n,n)&=(W_kx_n)e^{in\theta}\\
g(x_m,x_n,m-n)&=\text{Re}[(W_qx_m)(W_kx_n)^*e^{i(m-n)\theta}]
\end{aligned}
$$
其中 $\text{Re}[\cdot]$ 表示负数的实部，$(\cdot)^*$ 表示 $\cdot$ 的共轭复数，变换 $f_{q,k}$ 可以由以下矩阵乘法表示：
$$
\begin{aligned}
f_{\{q,k\}}(x_m,m)&=
\begin{pmatrix}
\cos m\theta & -\sin m\theta\\
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
W_{\{q,k\}}^{(11)} & W_{\{q,k\}}^{(12)}\\
W_{\{q,k\}}^{(21)} & W_{\{q,k\}}^{(22)}
\end{pmatrix}
\begin{pmatrix}
x_m^{(1)}\\
x_m^{(2)}
\end{pmatrix}\\
&=\begin{pmatrix}
\cos m\theta & -\sin m\theta\\
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
\{q,k\}_m^{(1)}\\
\{q,k\}_m^{(2)}
\end{pmatrix}
\end{aligned}
$$

**提示**：从复数的角度出发，任何 2D 平面上的向量均可以表示为 $z=a+bi$，同时根据欧拉公式

$$
e^{ix}=\cos x + i\cdot\sin x
$$

所以  $e^{im\theta}=\cos m\theta + i\cdot \sin m\theta$，那么：

$$
\begin{aligned}
(a+bi)e^{im\theta}
&=(a+bi)(\cos m\theta + i \sin m\theta)\\
&=\underbrace{(a\cos m\theta -b\sin m\theta)}_{\text{Real}} + i\cdot\underbrace{(b\cos m\theta + a\sin m\theta)}_{\text{Imaginary}}
\end{aligned}
$$

写成矩阵乘法的形式：

$$
\begin{pmatrix}
\cos m\theta & -\sin m\theta\\
\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
a \\
b
\end{pmatrix}
$$

因此乘 $e^{im\theta}$，相当于将 2D 向量 $W_qx_m$ 在平面中旋转 $m\theta$ 角度。

$$
\begin{aligned}
q^\mathrm{T}k&=f_q^\mathrm{T}(x_m,m)f_k(x_n,n)\\
&=
\begin{pmatrix}
q_m^{(1)} & q_m^{(2)}
\end{pmatrix}
\begin{pmatrix}
\cos m\theta & \sin m\theta\\
-\sin m\theta & \cos m\theta
\end{pmatrix}
\begin{pmatrix}
\cos n\theta & -\sin n\theta\\
\sin n\theta & \cos n\theta
\end{pmatrix}
\begin{pmatrix}
k_m^{(1)}\\
k_m^{(2)}
\end{pmatrix}\\
&=
\begin{pmatrix}
q_m^{(1)} & q_m^{(2)}
\end{pmatrix}
\begin{pmatrix}
\cos m\theta\sin n\theta + \sin m\theta\sin n\theta & \sin m\theta \cos n\theta-\cos m\theta \sin n\theta\\
-\sin m\theta \cos n\theta+\cos m\theta \sin n\theta & \cos m\theta\sin n\theta + \sin m\theta\sin n\theta
\end{pmatrix}
\begin{pmatrix}
k_m^{(1)}\\
k_m^{(2)}
\end{pmatrix}\\
&=
\begin{pmatrix}
q_m^{(1)} & q_m^{(2)}
\end{pmatrix}
\begin{pmatrix}
\cos (n-m)\theta & -\sin (n-m)\theta\\
\sin (n-m)\theta & \cos (n-m)\theta
\end{pmatrix}
\begin{pmatrix}
k_m^{(1)}\\
k_m^{(2)}
\end{pmatrix}\\
&=(W_qx_m)^\mathrm{T}R_{\Theta,n-m}^2(W_kx_n)
\end{aligned}
$$

### 📝通用形式

RoPE 可以从 2D 的简单场景扩展到高维场景，考虑旋转矩阵，将 d 维切分为 $d/2$ 个 2d 子空间，每个子空间对应一个正交旋转矩阵，$R_{\Theta,n-m}^d=(R_{\Theta,m}^d)^\mathrm{T}R_{\Theta,n}^d$.
$$
R_{\Theta,m}^d=\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0\\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0\\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0\\
\vdots & \vdots & \vdots & \vdots & & \vdots & \vdots\\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2}\\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}\\
\end{pmatrix}
$$
其中 $\theta_i= 1 / \text{base}^{2i/d}$，base 通常取 $10000$，$m$ 为 token 在序列的所在位置。那么 queries 和 keys  的内积（注意力）计算为：
$$
q^\mathrm{T}k=\underbrace{(W_qx_m)^\mathrm{T}R_{\Theta,n-m}^d(W_kx_n)}_{g(x_m,x_n,n-m)}
$$
**注意**：RoPE 的位置编码方式是乘性的（multiplicative），不是加性的（additive）。



### ⚙️高效计算方式

由于 RoPE 的旋转矩阵是一个系数矩阵，直接使用矩阵乘法，即 $R_{\Theta,m}^d x$ 的方式注入 RoPE 位置编码计算低效，一种高效的等价做法是：

$$
R_{\Theta,m}^d x=
\begin{pmatrix}
x_1\\
x_2\\
x_3\\
x_4\\
\vdots\\
x_{d-1}\\
x_d
\end{pmatrix}
\otimes
\begin{pmatrix}
\cos m\theta_1\\
\cos m\theta_1\\
\cos m\theta_2\\
\cos m\theta_2\\
\vdots\\
\cos m\theta_{d/2}\\
\cos m\theta_{d/2}
\end{pmatrix}
+\begin{pmatrix}
-x_2\\
x_1\\
-x_4\\
x_3\\
\vdots\\
-x_d\\
x_{d-1}
\end{pmatrix}
\otimes
\begin{pmatrix}
\sin m\theta_1\\
\sin m\theta_1\\
\sin m\theta_2\\
\sin m\theta_2\\
\vdots\\
\sin m\theta_{d/2}\\
\sin m\theta_{d/2}
\end{pmatrix}
$$

其中 $\otimes$ 表示按元素相乘（Element-wise Product），**证明方法就是将矩阵x向量展开**。



> [!NOTE]
>
> 在实际的 Qwen、Llama 等 LLMs 中，为了计算高效，采用了一种数学上等价的实现（不是相邻两个维度的配对），即将 $x_i$ 与 $x_{i+d/2}$ 进行配对，，对应的 `cos` 和 `sin` 向量也需要进行调整：
> $$
> R_{\Theta,m}^d x=
> \begin{pmatrix}
> x_1\\
> x_2\\
> x_3\\
> x_4\\
> \vdots\\
> x_{d-1}\\
> x_d
> \end{pmatrix}
> \otimes
> \begin{pmatrix}
> \cos m\theta_1\\
> \cos m\theta_2\\
> \cos m\theta_3\\
> \cos m\theta_4\\
> \vdots\\
> \cos m\theta_{d/2-1}\\
> \cos m\theta_{d/2}
> \end{pmatrix}
> +\begin{pmatrix}
> -x_2\\
> x_1\\
> -x_4\\
> x_3\\
> \vdots\\
> -x_d\\
> x_{d-1}
> \end{pmatrix}
> \otimes
> \begin{pmatrix}
> \sin m\theta_1\\
> \sin m\theta_2\\
> \sin m\theta_3\\
> \sin m\theta_4\\
> \vdots\\
> \sin m\theta_{d/2-1}\\
> \sin m\theta_{d/2}
> \end{pmatrix}
> $$
>
>
> ```python
> def rotate_half(x):
>  x1 = x[..., :x.shape[-1] // 2]
>  x2 = x[..., x.shape[-1] // 2:]
>  torch.cat((-x2, x1), dim=-1)
> 
> def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim = 1):
>  cos = cos.unsqueeze(unsqueeze_dim)
>  sin = sin.unsqueeze(unsqueeze_dim)
>  q_embed = q * cos + (rotate_half(q) * sin)
>  k_embed = k * cos + (rotate_half(k) * sin)
>  return q_embed, k_embed
> ```



## 多模态位置编码

多模态位置编码由 Qwen2-VL、Qwen2.5-VL 引入，由于 1D RoPE（适用于纯文本）无法编码**图像的空间位置、视频的时空位置**，不适用于多模态场景，因此需要专门设计一种多模态位置编码，具体来说，MRoPE 将维度 $d$ 切分为 3 个部分，分别编码：时间（temporal）、高度（height）、宽度（width）三个维度的位置信息。

MRoPE 的位置编码的形状为：$(3,\mathbf{B},\mathbf{S})$，其中 $\mathbf{B}$ 为 batch size，$\mathbf{S}$ 为序列长度，分为三种情况（设开始位置为 0、高 $H=2$、宽 $W=3$、视频帧数为 $T=2$）：

- **纯文本**：三个分量是一致的， $t=h=w=[0,1,...]$，此时 MRoPE 退化为 1D RoPE。
- **视频**：
  - $t=[0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2]$
  - $h=[0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1]$
  - $w=[0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2]$
- **图片**：时间维度的编码值相等
  - $t=[0,0,0,0,0,0]$
  - $h=[0,0,0,1,1,1]$
  - $w=[0,1,2,0,1,2]$

![RoPE](/tech-blog/images/Deep-Learning/RoPE/MRoPE.png)

## 🧑‍💻PyTorch 实现代码



### 绝对位置编码

首先，基于 PyTorch 实现原始 Transformer 使用的绝对位置编码：

$$
\text{PE}_{2i}(pos)=\sin(\frac{pos}{10000^{2i/d_{\text{model}}}}),\quad \text{PE}_{2i+1}(pos)=\cos(\frac{pos}{10000^{2i/d_{\text{model}}}})
$$

实现思路：输入为 `d_model`、`max_seq_len`、`theta`，实现 `init_pos_embs` 函数注册位置编码缓存（注意设置 `persistent=False` 避免将位置编码保存到模型的状态字典中）

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10000.0):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.init_pos_embs()

    def forward(self, x: torch.Tensor):
        # Apply transformer sinusoidal embedding.
        # (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(dtype=x.dtype)

    def init_pos_embs(self):
        pos = torch.arange(self.max_seq_len, dtype=torch.float32)  # (max_seq_len,)
        _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float32)
        inv_freqs = 1 / self.theta ** (_2i / self.d_model)
        args = pos[:, None] * inv_freqs[None, :]
        pe = torch.empty(self.max_seq_len, self.d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(args)
        pe[:, 1::2] = torch.cos(args)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
```



### 旋转位置编码（1D）

RoPE 为乘性位置编码，应用于 Self-Attention 模块中 Q、K，为了实现高效，将 $x_i$ 和 $x_{i+d/2}$ 进行配对，**注意**：在 `head_dim` 维度上应用 RoPE。

```python
class RoPE1D(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int, theta: float = 10000.0):
        super(RoPE1D, self).__init__()
        self.d_head = d_head
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.init_pos_embs()

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.size(-2)
        q_cos = self.cos[:, :, :seq_len].to(q.dtype)
        q_sin = self.sin[:, :, :seq_len].to(q.dtype)
        q = self.apply_rotary_pos_emb(q, q_cos, q_sin)

        k_cos = self.cos[:, :, :seq_len].to(k.dtype)
        k_sin = self.sin[:, :, :seq_len].to(k.dtype)
        k = self.apply_rotary_pos_emb(k, k_cos, k_sin)

        return q, k

    def rotate_half(self, x: torch.Tensor):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        return x * cos + self.rotate_half(x) * sin

    def init_pos_embs(self):
        pos = torch.arange(0, self.max_seq_len, dtype=torch.float32)
        _2i = torch.arange(0, self.d_head, step=2, dtype=torch.float32)
        inv_freqs = 1 / self.theta ** (_2i / self.d_head)
        args = pos[:, None] * inv_freqs[None, :]  # (max_seq_len, d_model / 2)
        args = torch.cat([args, args], dim=1)  # (max_seq_len, d_model)
        self.register_buffer("cos", torch.cos(args)[None, None, :, :], persistent=False)
        self.register_buffer("sin", torch.sin(args)[None, None, :, :], persistent=False)

```

### 🔥多模态旋转位置编码（3D）

3D RoPE 由 Qwen-VL 系列引入，适合编码图像、视频等多模态数据，将 `head_dim` 分为 3 个均匀的部分，分别编码：时间、高度、宽度的位置信息。

**TODO**：给出一个简化实现...

## 参考文献

[1] Vaswani et. al., Attention Is All You Need, arxiv:1706.03762, 2017

[2] Jianlin Su et. al.,RoFormer: Enhanced Transformer with Rotary Position Embedding, 2023

[3] Team Qwen, Qwen2-VL Technical Report, arXiv:2409.12191, 2024

[4] Team Qwen, Qwen2.5-VL Technical Report, arXiv:2502.13923, 2025
