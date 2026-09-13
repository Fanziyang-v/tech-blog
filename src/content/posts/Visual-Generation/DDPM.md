---
title: "去噪扩散概率模型（DDPM）"
description: "📝动手推导 DDPM 公式"
date: 2026-09-13
tags: ["深度学习", "扩散模型", "多模态生成"]
draft: false

---

## Evidence Lower Bound (ELBO)

假设 $x$ 表示真实数据，其分布为 $p(x)$；$z$ 表示潜变量，其分布为 $q_\phi(z|x)$，联合概率分布为 $p(x, z)$，则证据下界（Evidence Lower Bound, ELBO）为：

$$
\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]
$$

ELBO 是优化的代理目标🎯，跟证据 $\log p(x)$ 有如下不等关系：

$$
\log p(x) \ge \mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]
$$

🤗**详细推导过程**：基于两个基本公式，即边缘概率和条件概率公式。

1. **边缘概率公式**：

$$
p(x)=\int p(x,z) dz
$$

2. [**条件概率公式**](https://en.wikipedia.org/wiki/Chain_rule_(probability))：

$$
p(x)=\frac{p(x,z)}{p(z|x)}
$$

从公式 (3) 开始推导：

$$
\begin{aligned}
\log p(x)&=\log\int p(x,z) dz\\
&=\log\int \frac{p(x,z)q_\phi(z|x)}{q_\phi(z|x)} dz\\
&=\log \mathbb{E}_{q_\phi(z|x)}[\frac{p(x,z)}{q_\phi(z|x)}]\\
&\ge \mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]
\end{aligned}
$$

> [!NOTE]
>
> 这里用到了 [Jensen 不等式](https://en.wikipedia.org/wiki/Jensen%27s_inequality)。

从公式 (4) 开始推导：

$$
\begin{aligned}
\log p(x)&=\log p(x) \int q_\phi(z|x)dz\\
&=\int (\log p(x)) q_\phi(z|x)dz\\
&=\mathbb{E}_{q_\phi(z|x)}[\log p(x)]\\
&=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{p(z|x)}]\\
&=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)q_\phi(z|x)}{p(z|x)q_\phi(z|x)}]\\
&=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]+\mathbb{E}_{q_\phi(z|x)}[\frac{q_\phi(z|x)}{p(z|x)}]\\
&=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]+ D_{KL}(q_\phi(z|x)\Vert p(z|x))\\
&\ge \mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]
\end{aligned}
$$

> [!NOTE]
>
> KL 散度始终大于 0

💡从这个推导过程中，可以发现一个**关键等式**：

$$
\log p(x)=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]
+ D_{KL}(q_\phi(z|x)\Vert p(z|x))
$$

因此，最大化 $\log p(x)$ 等价于最大化右侧的两项和，等价于最小化 $D_{KL}(q_\phi(z|x)\Vert p(z|x))$ 或最大化 ELBO 。

## Variational AutoEncoder (VAE)

变分自编码器（Variational AutoEncoder, VAE）包含两个组件：

1. 编码器 $q_\phi(z|x)$：将真实数据转换为潜空间表示；
2. 解码器 $p_\theta(x|z)$：将潜变量还原回真实数据。

通常情况下，潜变量 $z$ 的维度小于真实数据 $x$ 的维度，可用于特征降维，加速 Diffusion 模型的训练和推理速度（如 Stable Diffusion、Wan 2.2）。

<center><img src="/tech-blog/images/Visual-Generation/DDPM/VAE.png" alt="VAE" style="zoom: 50%;"></center>

默认情况下，变分自编码器（Variational AutoEncoder, VAE） 直接优化 ELBO，将 ELBO 项进一步拆分：

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(x,z)}{q_\phi(z|x)}]
&=\mathbb{E}_{q_\phi(z|x)}[\log \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)}]\\
&=\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]+\mathbb{E}_{q_\phi(z|x)}[\log \frac{p(z)}{q_\phi(z|x)}]\\
&=\underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction term}}-\underbrace{D_{KL}(q_\phi(z|x)\Vert p(z))}_{\text{prior matching term}}
\end{aligned}
$$

> [!NOTE]
>
> 第一项（重建项）表示从潜变量还原为真实数据的可能性，第二项（先验匹配项）表示潜变量分布和先验分布的相似性。

VAE 的编码器通常建模一个**多元高斯分布**，通常选择标准多元高斯分布：

$$
\begin{aligned}
q_\phi(z|x)&=\mathcal{N}(z;\mu_\phi(x),\sigma^2_\phi (x)\mathbf{I})\\
p(z)&=\mathcal{N}(z;\mathbf{0},\mathbf{I})
\end{aligned}
$$

ELBO 的 KL 散度项可以通过解析求解，重建项可以通过蒙特卡洛估计（Monte Carlo estimate）求解：

$$
\arg \max_{\phi,\theta} \mathbb{E}_{q_\phi(x|z)}[\log p_\theta(x|z)]-D_{KL}(q_\phi(z|x)\Vert p(z))\approx \arg \max_{\phi,\theta} \sum_{l=1}^L \log p_\theta(x|z^{(l)}) -D_{KL}(q_\phi(z|x)\Vert p(z))
$$

其中$z^{(l)} \sim q_\phi(z|x)$，这里存在一个小问题 🧐：由于 $z^{(l)}$ 是随机采样的，导致这个损失函数**不可微**。若 $q_\phi(z|x)$ 被建模为一种确定性分布，那么可以通过**重参数化技巧**（Reparameterization trick）解决。

具体来说，**重参数化技巧就是将一个随机变量写为一个噪声变量的决定性函数，使非随机项可以通过梯度下降优化**，假设 $q_\phi(z|x)$ 被建模为多元高斯分布，即 $x\sim \mathcal{N}(x;\mu,\sigma^2)$，那么可以将一个随机变量重写为：

$$
x=\mu+\sigma \epsilon
$$

其中 $\epsilon \sim \mathcal{N}(\epsilon;0,I)$ 为标准高斯分布，因此在 VAE 中有：

$$
z=\mu_\phi(x)+\sigma_\phi(x) \odot \epsilon
$$

这里 $\epsilon \sim \mathcal{N}(\epsilon;\mathbf{0},\mathbf{I})$  为标准多元高斯分布，$\odot$ 表示逐元素相乘。

## Hierarchical Variational AutoEncoder (HVAE)

层级变分自编码器（Hierarchical Variational AutoEncoder, HVAE）是 VAE 的泛化版本，包含多个潜变量，结构如下图所示：

<center><img src="/tech-blog/images/Visual-Generation/DDPM/HVAE.png" alt="HVAE" style="zoom: 67%;"></center>

虽然通常情况下 HVAE 中的每个潜变量 $z_t$ 基于前面所有的潜变量 $z_i, i=1,2,\dots,t-1$，但是这里只需要考虑一个特殊情况，即 MHVAE，生成过程是一个**马尔可夫链**，每个变换仅基于相邻的潜变量，假设有 $T$ 个层级，那么有：

$$
\begin{aligned}
p(x,z_{1:T})&=p(z_T)p_\theta(x|z_1)\prod_{t=2}^T p(z_{t-1}|z_t)\\
q_\phi(z_T|x)&=q_\phi(z_1|x)\prod_{t=2}^Tq_\phi(z_t|z_{t-1})
\end{aligned}
$$

与此同时，ELBO 项可以扩展到 HVAE 中（有类似的形式）：

$$
\begin{aligned}
\log p(x)&=\log \int p(x,z_{1:T}) dz_{1:T}\\
&=\log \int \frac{q_\phi(z_T|x)p(x,z_{1:T})}{q_\phi(z_T|x)} dz_{1:T}\\
&=\log \mathbb{E}_{q_\phi(z_T|x)}[\frac{p(x,z_{1:T})}{q_\phi(z_T|x)}]\\
&\ge \mathbb{E}_{q_\phi(z_T|x)}[\log \frac{p(x,z_{1:T})}{q_\phi(z_T|x)}]
\end{aligned}
$$

将上述两个等式代入 ELBO 项中，重写 ELBO 得到：

$$
\begin{aligned}
\mathbb{E}_{q_\phi(z_T|x)}[\log \frac{p(x,z_{1:T})}{q_\phi(z_T|x)}]
&=\mathbb{E}_{q_\phi(z_T|x)}[\log \frac{p(z_T)p_\theta(x|z_1)\prod_{t=2}^T p(z_{t-1}|z_t)}{q_\phi(z_1|x)\prod_{t=2}^Tq_\phi(z_t|z_{t-1})}]
\end{aligned}
$$

## Variational Diffusion Models (VDMs)

变分扩散模型（Variational Diffusion Models, VDMs）可以视为一种 MHVAE，有三个限制条件：

1. 潜空间维度**等于**真实数据维度
2. 编码器被建模为**线性高斯模型**，不包含任何可学习参数。
3. 每个时间步 $t$ 编码器的高斯分布参数随着时间变化，**最后一个时间步 $T$ 为标准多元高斯分布**。

为了方便公式推导，统一记号，将真实数据记为 $x_0$，$x_t, t\in [1,T]$ 为第 $t$ 步的潜变量。

基于**第一个**假设：

$$
q(x_{1:T}|x_0)=\prod_{t=1}^T q(x_t|x_{t-1})
$$

基于**第二个**假设，将高斯编码器参数化为：

$$
\mu_t(x_t)=\sqrt{\alpha_t}x_{t-1}, \quad \Sigma_t(x_t)=(1-\alpha_t)\mathbf{I}
$$

编码器变化可以写为：

$$
q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)\mathbf{I})
$$

基于第三个假设，基于 MHVAE 的推导，联合概率分布可以重写为：

$$
p(x_{0:T})=p(x_T)\prod_{t=1}^T p(x_{t-1}|x_t)
$$

其中 $p(x_T)=\mathcal{N}(x_T;\mathbf{0},\mathbf{I})$，Diffusion 的前向过程可以理解为逐渐给一个图片添加高斯噪声，直到最后成为一个完整的高斯噪声，该过程如下图所示。

<center><img src="/tech-blog/images/Visual-Generation/DDPM/VDM.png" alt="VDM" style="zoom: 67%;"></center>

**VDMs 的 ELBO 详细推导过程**：

$$
\begin{aligned}
\log p(x)&=\log \int p(x_{0:T}) dx_{1:T}\\
&=\log \int \frac{p(x_{0:T})q(x_{1:T}|x_0)}{q(x_{1:T}|x_0)} dx_{1:T}\\
&=\log \mathbb{E}_{q(x_{1:T}|x_0)}[\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}]\\
&\ge \mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^T q(x_t|x_{t-1})}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)\prod_{t=2}^T p_\theta(x_{t-1}|x_t)}{q(x_T|x_{T-1})\prod_{t=1}^{T-1} q(x_t|x_{t-1})}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)\prod_{t=1}^{T-1} p_\theta(x_{t}|x_{t+1})}{q(x_T|x_{T-1})\prod_{t=1}^{T-1} q(x_t|x_{t-1})}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log \frac{p(x_T)p_\theta(x_0|x_1)}{q(x_T|x_{T-1})}] + \mathbb{E}_{q(x_{1:T}|x_0)}[\log \prod_{t=1}^{T-1} \frac{p_\theta(x_|x_{t+1})}{q(x_t|x_{t-1})}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log p_\theta(x_0|x_1)]+\mathbb{E}_{q(x_{1:T}|x_0)}[\log \frac{p(x_T)}{q(x_T|x_{T-1})}]+\mathbb{E}_{q(x_{1:T}|x_0)}[\sum_{t=1}^{T-1} \log\frac{p_\theta(x_|x_{t+1})}{q(x_t|x_{t-1})}]\\
&=\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]+\mathbb{E}_{q(x_{T-1},x_T|x_0)}[\log \frac{p(x_T)}{q(x_T|x_{T-1})}]+\sum_{t=1}^{T-1}\mathbb{E}_{q(x_{t-1},x_t,x_{t+1}|x_0)}[ \log\frac{p_\theta(x_t|x_{t+1})}{q(x_t|x_{t-1})}]\\
&=\underbrace{\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]}_{\text{reconstruction term}}
-\underbrace{\mathbb{E}_{q(x_{T-1}|x_0)}[D_{KL}(q(x_T|x_{T-1})\Vert p(x_T))}_{\text{prior matching term}}
-\sum_{t=1}^{T-1}\underbrace{\mathbb{E}_{q(x_{t-1},x_{t+1}|x_0)}[D_{KL}(q(x_t|x_{t-1}) \Vert p_\theta(x_t|x_{t+1}))]}_{\text{consistency term}}\\
\end{aligned}
$$

上述 ELBO 推导三个部分组成，分别具有清晰的含义：

1. $\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]$ 为**重建项**，表示从 $x_1$ 恢复为 $x_0$ 的对数概率
2. $\mathbb{E}_{q(x_{T-1}|x_0)}[D_{KL}(q(x_T|x_{T-1})\Vert p(x_T))$ 为**先验匹配项**，表示从 $x_{T-1}$ 到 $x_T$ 的近似先验分布与真实先验分布（标准多元高斯分布）的**匹配度**/ KL 散度，通常情况下，可以假设若 $T$ **足够大**（例如 $T=1000$）那么 $q(x_T|x_{T-1})$ 为标准多元高斯分布，这一项**可近似为 0**。
3. $\mathbb{E}_{q(x_{t-1},x_{t+1}|x_0)}[D_{KL}(q(x_t|x_{t-1}) \Vert p_\theta(x_t|x_{t+1}))]$ 为**一致项**，衡量从**前向和后向**两个过程到达 $x_t$ 时的**分布一致性**，这一项可以通过让分布 $p_\theta(x_t|x_{t+1})$ 与 $q(x_t|x_{t-1})$ 分布接近（KL 散度趋于 0）来**最小化**。

<center><img src="/tech-blog/images/Visual-Generation/DDPM/VDM2.png" alt="VDM" style="zoom: 67%;"></center>

这个推导结果可以通过蒙特卡洛估计的方式进行优化，但这种优化并不是最优的，因为**一致项**是在两个随机变量 $\{x_{t-1},x_{t+1}\}$ 下计算的期望（$T - 1$ 个一致项求和结果），蒙特卡洛估计的方差会高于仅通过一个随机变量估计的一致性项，因此下面推导一个**更好的目标函数**🎯。

推导的核心在于可以**引入一个冗余条件项**，每个时间步编码器的转换可以重写为：

$$
q(x_t|x_{t-1},x_0)=\frac{q(x_t,x_{t-1},x_0)}{q(x_{t-1},x_0)}=\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)\cancel{q(x_0)}}{q(x_{t-1}|x_0)\cancel{q(x_0)}}=\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}
$$

重新推导 ELBO 项：

$$
\begin{aligned}
\log p(x) &\ge \mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)}{\prod_{t=1}^T q(x_t|x_{t-1},x_0)}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)\prod_{t=2}^T p_\theta(x_{t-1}|x_t)}{q(x_1|x_0)\prod_{t=2}^T q(x_t|x_{t-1},x_0)}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)}{q(x_1|x_0)}+\log\prod_{t=2}^T\frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1},x_0)}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)}{q(x_1|x_0)}+\log\prod_{t=2}^T\frac{p_\theta(x_{t-1}|x_t)}{\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)}{q(x_{t-1}|x_0)}}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)}{q(x_1|x_0)}+\log \prod_{t=2}^T\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}+\log \prod_{t=2}^T\frac{q(x_{t-1}|x_0)}{q(x_t|x_0)}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)}{\cancel{q(x_1|x_0)}}+\log \prod_{t=2}^T\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}+\log \frac{\cancel{q(x_1|x_0)}}{q(x_T|x_0)}]\\
&=\mathbb{E}_{q(x_{1:T}|x_0)}[\log\frac{p(x_T)p_\theta(x_0|x_1)}{q(x_T|x_0)}+\log \prod_{t=2}^T\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}]\\
&=\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]+\mathbb{E}_{q(x_T|x_0)}[\log\frac{p(x_T)}{q(x_T|x_0)}]+\sum_{t=2}^T\mathbb{E}_{q(x_{t-1},x_t|x_0)}[\frac{p_\theta(x_{t-1}|x_t)}{q(x_{t-1}|x_t,x_0)}]\\
&=\underbrace{\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]}_{\text{reconstruction term}}
-
\underbrace{D_{KL}(q(x_T|x_0)\Vert p(x_T))}_{\text{prior matching term}}
-\sum_{t=2}^T \underbrace{\mathbb{E}_{q(x_t|x_0)} D_{KL}(q(x_{t-1}|x_t,x_0)\Vert p_\theta(x_{t-1}|x_t))}_{\text{denoising matching term}}\\
\end{aligned}
$$

上述推导得到了一个方差更小，更易于优化的 ELBO 目标，分别具有特别含义：

1. $\mathbb{E}_{q(x_1|x_0)}[\log p_\theta(x_0|x_1)]$ 为**重建项**，表示从 $x_1$ 恢复为 $x_0$ 的对数概率，可以通过蒙特卡洛估计进行优化
2. $D_{KL}(q(x_T|x_0)\Vert p(x_T))$ 为**先验匹配项**，表示从 $x_{T-1}$ 到 $x_T$ 的近似先验分布与真实先验分布（标准多元高斯分布）的**匹配度**/ KL 散度，通常情况下，可以假设若 $T$ **足够大**（例如 $T=1000$）那么 $q(x_T|x_{T-1})$ 为标准多元高斯分布，这一项**可近似为 0**。
3. $\mathbb{E}_{q(x_t|x_0)} D_{KL}(q(x_{t-1}|x_t,x_0)\Vert p_\theta(x_{t-1}|x_t))$ 为**去噪匹配项**，$q(x_{t-1}|x_t,x_0)$ 可以视为 Ground Truth，优化去噪模型参数 $\theta$，使得去噪后的分布和添加高斯噪声后在 $x_{t-1}$ 的分布一致，这一项通过**最小化 KL 散度**进行优化。

<center><img src="/tech-blog/images/Visual-Generation/DDPM/VDM3.png" alt="VDM" style="zoom: 67%;"></center>

由贝叶斯公式，可以进行如下转换：

$$
\begin{aligned}
q(x_{t-1}|x_t,x_0)&=\frac{q(x_t,x_{t-1},x_0)}{q(x_t,x_0)}=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)\cancel{q(x_0)}}{q(x_t|x_0)\cancel{q(x_0)}}=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}
\end{aligned}
$$

下面开始推导**一步加噪公式**，由于 $q(x_t|x_{t-1},x_0)=q(x_t|x_{t-1})=\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)\mathbf{I})$，因此有如下公式：

$$
x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t
$$

其中 $\epsilon_t \sim \mathcal{N}(\epsilon_t;\mathbf{0},\mathbf{I})$，类似地对于 $x_{t-1} \sim q(x_{t-1}|x_{t-2})$ 有如下公式：

$$
x_{t-1}=\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon
$$

基于这 $t$ 个等式进行化简：

$$
\begin{aligned}
x_t&=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\\
&=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-1})+\sqrt{1-\alpha_t}\epsilon_t\\
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{\alpha_t(1-\alpha_{t-1})}\epsilon_{t-1}+\sqrt{1-\alpha_t}\epsilon_t\\
&=\sqrt{\alpha_t\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_t\alpha_{t-1}}\epsilon_{t-2}\\
&=\cdots\\
&=\sqrt{\alpha_t\alpha_{t-1}\cdots\alpha_1}x_0+\sqrt{1-\alpha_t\alpha_{t-1}\cdots\alpha_1}\epsilon_0\\
&=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon
\end{aligned}
$$

其中 $\bar{\alpha}_t=\prod_{t=1}^T\alpha_t$，因此得到**一次加噪**公式：$x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon$。


> [!note]
> 这里用到了独立同分布的**正态分布可加性**，若 $X\sim \mathcal{N}(\mu_1,\sigma_1^2),Y\sim \mathcal{N}(\mu_2,\sigma_2^2)$，则 $aX+bY$ 也为正态分布，且 $aX+bY\sim \mathcal{N}(a\mu_1+b\mu_2,a^2\sigma_1^2+b^2\sigma_2^2)$

因此，$q(x_{t-1}|x_0)=\mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})\mathbf{I})$、$q(x_t|x_0)=\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)\mathbf{I})$，因此：

$$
\begin{aligned}
q(x_{t-1}|x_t,x_0)&=\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)}\\
&=\frac{\mathcal{N}(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)\mathbf{I})\mathcal{N}(x_{t-1};\sqrt{\bar{\alpha}_{t-1}}x_0,(1-\bar{\alpha}_{t-1})\mathbf{I})}{\mathcal{N}(x_t;\sqrt{\bar{\alpha}_t}x_0,(1-\bar{\alpha}_t)\mathbf{I})}\\
&\propto \exp\{-\frac{1}{2}[\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{1-\alpha_t}+\frac{(x_{t-1}-\sqrt{\bar{\alpha}_{t-1}}x_0)^2}{1-\bar{\alpha}_{t-1}}-\frac{(x_t-\sqrt{\bar{\alpha}_t}x_0)^2}{1-\bar{\alpha}_t}] \}\\
&=\exp \{-\frac{1}{2}[\frac{(\alpha_tx_{t-1}^2-2\sqrt{\alpha_t}x_tx_{t-1})}{1-\alpha_t}]+\frac{(x_{t-1}^2-2\sqrt{\bar{\alpha}_{t-1}}x_{t-1}x_0)}{1-\bar{\alpha}_{t-1}} +C(x_t,x_0) \}\\
&\propto \exp \{ -\frac{1}{2}[\frac{(\alpha_tx_{t-1}^2-2\sqrt{\alpha_t}x_tx_{t-1})}{1-\alpha_t}]+\frac{(x_{t-1}^2-2\sqrt{\bar{\alpha}_{t-1}}x_{t-1}x_0)}{1-\bar{\alpha}_{t-1}} \}\\
&=\exp \{ -\frac{1}{2}[\frac{\alpha_t}{1-\alpha_t}x_{t-1}^2-\frac{2\sqrt{\alpha_t}}{1-\alpha_t}x_tx_{t-1}+\frac{1}{1-\bar{\alpha}_{t-1}}x_{t-1}^2-\frac{2\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_{t-1}x_0] \}\\
&=\exp \{ -\frac{1}{2}[(\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\bar{\alpha}_{t-1}})x_{t-1}^2-2(\frac{\sqrt{\alpha_t}}{1-\alpha_t}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0)x_{t-1}] \}\\
&=\exp \{ -\frac{1}{2(\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\bar{\alpha}_{t-1}})} [x_{t-1}^2-2(\frac{\frac{\sqrt{\alpha_t}}{1-\alpha_t}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{1-\bar{\alpha}_{t-1}}x_0}{\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\bar{\alpha}_{t-1}}})] \}\\
&=\exp \{ -\frac{1}{2(\frac{1-\alpha_t\bar{\alpha}_{t-1}}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})})}[x_{t-1}^2-2\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\alpha_t\bar{\alpha}_{t-1}}x_{t-1}] \}\\
&=\exp \{ -\frac{1}{2(\frac{1-\bar{\alpha}_t}{(1-\alpha_t)(1-\bar{\alpha}_{t-1})})}[x_{t-1}^2-2\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t}x_{t-1}] \}\\
&\propto \mathcal{N}(x_{t-1};\underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t}}_{\mu_q(x_t,x_0)},\underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}}_{\Sigma_q(t)})
\end{aligned}
$$

这里简记 $\Sigma_q(t)=\sigma_q^2(t)\mathbf{I}$，方便后续公式推导。

> [!NOTE]
>
> 上面这个公式推导时 $x_0$ 和 $x_t$ 可以视为常数。

根据[两个多元高斯分布的 KL 散度公式](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions)：

$$
D_{KL}(\mathcal{x;\mu_x,\Sigma_x}\Vert \mathcal{y;\mu_y,\Sigma_y})=\frac{1}{2}[\log \frac{|\Sigma_y|}{|\Sigma_x|}-d+\text{tr}(\Sigma_y^{-1}\Sigma_x)+(\mu_y-\mu_x)^T\Sigma_y^{-1}(\mu_y-\mu_x)]
$$

因此：

$$
\begin{aligned}
&\arg \min_{\theta} D_{KL}(q_{t-1}|x_t,x_0)\Vert p_\theta(x_{t-1}|x_t)\\
=&\arg \min_{\theta} D_{KL}(\mathcal{N}(x_{t-1};\mu_q,\Sigma_q(t)))\Vert \mathcal{N}(x_{t-1};\mu_\theta,\Sigma_q(t))) \\
=&\arg \min_{\theta} \frac{1}{2}[\log \frac{|\Sigma_q(t)|}{|\Sigma_q(t)|}-d+\text{tr}(\Sigma_q(t)^{-1}\Sigma_q(t))+(\mu_\theta-\mu_q)^T\Sigma_q(t)^{-1}(\mu_\theta-\mu_q)]\\
=&\arg \min_{\theta} \frac{1}{2}[\log 1-d+d+(\mu_\theta-\mu_q)^T(\sigma^2_q(t)\mathbf{I})^{-1}(\mu_\theta-\mu_q)]\\
=&\arg \min_{\theta} \frac{1}{2\sigma^2_q(t)}[\Vert \mu_\theta-\mu_q \Vert^2_2]

\end{aligned}
$$

其中 $\mu_q$ 表示 $\mu_q(x_t,x_0)$，而 $\mu_\theta$ 表示 $\mu_\theta(x_t,t)$，换句话说，我们要让 $\mu_\theta(x_t,t)$ 尽可能接近 $\mu_q(x_t,x_0)$，这就是扩散模型的优化目标，回顾之前推导的公式：

$$
\mu_q(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t}
$$

设定 $\mu_\theta(x_t,t)$ 为如下形式，这里 $\hat{x}_\theta(x_t,t)$ 表示给定一个位于时间步 $t$ 的图片 $x_t$，去噪模型对 $x_{t-1}$ 的预测结果：

$$
\mu_\theta(x_t,t)=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\hat{x}_\theta(x_t,t)}{1-\bar{\alpha}_t}
$$

这里 $\hat{x}_\theta(x_t,t)$ 表示给定一个位于时间步 $t$ 的图片 $x_t$，去噪模型对 $x_{t-1}$ 的预测结果。将这两个等式代入方程进一步简化：

$$
\begin{aligned}
&\arg \min_{\theta} D_{KL}(q_{t-1}|x_t,x_0)\Vert p_\theta(x_{t-1}|x_t)\\
=&\arg \min_{\theta} \frac{1}{2\sigma^2_q(t)}[\Vert \mu_\theta(x_t,t)-\mu_q(x_t,x_0) \Vert^2_2]\\
=&\arg \min_{\theta} \frac{1}{2\sigma^2_q(t)}[\Vert  \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)\hat{x}_\theta(x_t,t)}{1-\bar{\alpha}_t}-\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t} \Vert^2_2]\\
=&\arg \min_{\theta} \frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{2\sigma^2_q(t)(1-\bar{\alpha}_t)^2}[\Vert \hat{x}_\theta(x_t,t)-x_0 \Vert^2_2]
\end{aligned}
$$


## $\epsilon$-Prediction format

在传统的 DDPM 里，模型预测的是噪声，本质上是等价的，由于 $x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon_0$，因此：

$$
x_0=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_0}{\sqrt{\bar{\alpha}_t}}
$$

与此同时， $\hat{x}_\theta(x_t,t)$ 可以重写（设定为与 $x_0$ 类似的形式）：

$$
\hat{x}_\theta(x_t,t)= \frac{x_t- \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}
$$

将这两个等式回代入方程 (32)，有：

$$
\begin{aligned}
&\arg \min_{\theta} D_{KL}(q_{t-1}|x_t,x_0)\Vert p_\theta(x_{t-1}|x_t)\\
=&\arg \min_{\theta} \frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{2\sigma^2_q(t)(1-\bar{\alpha}_t)^2}[\Vert \hat{x}_\theta(x_t,t)-x_0 \Vert^2_2]\\
=&\arg \min_{\theta} \frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{2\sigma^2_q(t)(1-\bar{\alpha}_t)^2}[\Vert \frac{x_t- \sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_\theta(x_t,t)}{\sqrt{\bar{\alpha}_t}}-\frac{x_t-\sqrt{1-\bar{\alpha}_t}\epsilon_0}{\sqrt{\bar{\alpha}_t}} \Vert_2^2]\\
=&\arg \min_{\theta} \frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{2\sigma^2_q(t)(1-\bar{\alpha}_t)^2}\frac{1-\bar{\alpha}_t}{\bar{\alpha}_t}[\Vert \epsilon_0 - \hat{\epsilon}_\theta(x_t,t) \Vert_2^2]\\
=&\arg \min_{\theta} \frac{\bar{\alpha}_{t-1}(1-\alpha_t)^2}{2\sigma^2_q(t)(1-\bar{\alpha}_t)\bar{\alpha}_t}[\Vert \epsilon_0 - \hat{\epsilon}_\theta(x_t,t) \Vert_2^2]
\end{aligned}
$$


## 🎨Sampling

上面分析了扩散模型的训练目标，下面推导采样公式，根据方程 (27)：

$$
q(x_{t-1}|x_t,x_0)\propto \mathcal{N}(x_{t-1};\underbrace{\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t}}_{\mu_q(x_t,x_0)},\underbrace{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{I}}_{\Sigma_q(t)})
$$

由于 sampling 过程是不知道原始图片 $x_0$ 的，因此可以用 $x_t$ 表示：

$$
x_t=\sqrt{\bar{\alpha}_t}x_0+\sqrt{1-\bar{\alpha}_t}\epsilon \Rightarrow x_0=\frac{1}{\sqrt{\bar{\alpha}_t}}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)
$$

将这个等式代入 $\mu_q(x_t,x_0)$ 中有：

$$
\begin{aligned}
\mu_q(x_t,x_0)&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})x_t+\sqrt{\bar{\alpha}_{t-1}}(1-\alpha_t)x_0}{1-\bar{\alpha}_t}\\

&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{\sqrt{\bar{\alpha}_{t-1}}}{\sqrt{\bar{\alpha}_t}}\frac{1-\alpha_t}{1-\bar{\alpha}_t}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)\\

&=\frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}x_t+\frac{1-\alpha_t}{\sqrt{\alpha}_t(1-\bar{\alpha}_t)}(x_t-\sqrt{1-\bar{\alpha}_t}\epsilon)\\

&=\frac{1}{\sqrt{\alpha_t}}[\frac{\alpha_t(1-\bar{\alpha}_{t-1})+(1-\alpha_t)}{1-\bar{\alpha}_t}x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon]\\

&=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon)\\

&=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))\\

\end{aligned}
$$

最后一步由于真实的噪声 $\epsilon$ 不可获得，因此使用去噪模型的预测噪声结果： $\epsilon_\theta(x_t,t)$ 替代，基于方程 (36)，可令：

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t,t))+\sigma_t\epsilon
$$

其中 $\sigma_t^2=\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}$ （**理论值**），但是在原始 DDPM 论文中，还尝试了一种设定即 $\sigma_t=\beta_t=1-\alpha_t$，**实际效果差不多**。

## 🔥Guidance

### Classifier Guidance

核心的思想是使用**预训练好的一个无条件扩散模型**，训练一个**图像分类器** $p_\phi(y|x_t)$，该图像分类器输入为带有噪声的图片，输出该图片为类别 y 的概率，在去噪过程中，利用分类器对噪声图像的梯度**引导去噪模型**向类别 y 靠拢。

$$
\nabla_{x_t}\log p(x_t|y)=\nabla_{x_t}\log\frac{p(y|x_t)p(x_t)}{p(y)}=\underbrace{\nabla_{x_t}\log p(x_t)}_{\text{unconditional score}}+\underbrace{\nabla_{x_t}\log p(y|x_t)}_{\text{adversarial gradient}}
$$


分类器对 $x_t$ 的梯度跟噪声 $\epsilon_\theta(x_t,t)$ 成反比，因此修正后：

$$
\widetilde{\epsilon}_\theta (x_t,t,y)=\epsilon_\theta(x_t,t)+s\nabla_{x_t}\log p(y|x_t)
$$

其中 $s$ 为引导强度（Guidance Scale）：

- 当 $s=0$ 时，等价于无条件生成；
- 当 $s>0$ 时，引导去噪模型向类别 y 靠拢，$s$ 越大矫正越强。

优点是**即插即用**，只需要有一个无条件图像生成器，然后再训练一个轻量级的噪声图片分类器就可以实现条件生成。

Classifier guidance 的缺点：

- **显著缺点**：需要额外训练一个图像分类器 $p_\phi(y|x_t)$，用于预测噪声图片 $x_t$ 属于类别 y 的概率，但相对于传统图像分类器，更难训练。
- **推理开销大**：去噪的每一步，不仅要运行去噪模型 $\epsilon_\theta(x_t,t)$，还要运行分类器并计算梯度。

### Classifier-free Guidance (CFG)

Classifier-free Guidance (CFG) 由 DDPM 作者 Jonathan Ho 等人提出，**不需要使用单独的噪声图像分类器**即可实现条件生成。

对方程 (42) 进行变换：

$$
\nabla_{x_t}\log p(y|x_t)=\underbrace{\nabla_{x_t}\log p(x_t|y)}_{\text{conditional score}}-\underbrace{\nabla_{x_t}\log p(x_t)}_{\text{unconditional score}}
$$

将变换后的等式代入方程 (43)，得到：

$$
\begin{aligned}
\widetilde{\epsilon}_\theta (x_t,t,y)&=\epsilon_\theta(x_t,t)+s(\nabla_{x_t}\log p(x_t|y)-\nabla_{x_t}\log p(x_t))\\
&=\epsilon_\theta(x_t,t,\phi)+s(\epsilon_\theta(x_t,t,y)-\epsilon_\theta(x_t,t,\phi))\\
&=(1-s)\cdot\epsilon_\theta(x_t,t,\phi)+s\cdot\epsilon_\theta(x_t,t,y)
\end{aligned}
$$

这里的 $s$ 为引导强度，跟 Classifier guidance 里面的含义是一致的，即 $s=0$ 等价于无条件生成，$s>0$ 时为条件生成且越大控制越强。

**优点**：CFG 不需要训练一个分类器 $p_\phi(y|x_t)$

**缺点**：仍然需要推理两次。

## 📜参考文献

[1] Jonathan Ho et al. Denoising Diffusion Probabilistic Models. UC Berkeley. 2020.

[2] Calvin Luo. Understanding Diffusion Models: A Unified Perspective. Google. 2022.

[3] Prafulla Dhariwal et al. Diffusion Models Beat GANs on Image Synthesis. OpenAI. 2021.

[4] Jonathan Ho et al. Classifier-free Diffusion Guidance. UC Berkeley. 2022.
