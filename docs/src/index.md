```@meta
CurrentModule = NormalizingFlows
```

# NormalizingFlows.jl

Documentation for [NormalizingFlows](https://github.com/TuringLang/NormalizingFlows.jl).


The purpose of this package is to provide a simple and flexible interface for 
variational inference (VI) and normalizing flows (NF) for Bayesian computation and generative modeling.
The key focus is to ensure modularity and extensibility, so that users can easily 
construct (e.g., define customized flow layers) and combine various components 
(e.g., choose different VI objectives or gradient estimates) 
for variational approximation of general target distributions, 
*without being tied to specific probabilistic programming frameworks or applications*. 

See the [documentation](https://turinglang.org/NormalizingFlows.jl/dev/) for more.  

## Installation
To install the package, run the following command in the Julia REPL:
```
]  # enter Pkg mode
(@v1.9) pkg> add git@github.com:TuringLang/NormalizingFlows.jl.git
```
Then simply run the following command to use the package:
```julia
using NormalizingFlows
```

## What are normalizing flows?

Normalizing flows transform a simple reference distribution $q_0$ (sometimes known as base distribution) to 
a complex distribution $q_\theta$ using invertible functions with trainable parameter $\theta$, aiming to approximate a target distribution $p$.
The approximation is achieved by minimizing some statistical distances between $q$ and $p$.

In more details, given the base distribution, usually a standard Gaussian distribution, i.e., $q_0 = \mathcal{N}(0, I)$,
we apply a series of parameterized invertible transformations (called flow layers), $T_{1, \theta_1}, \cdots, T_{N, \theta_k}$, yielding that
```math
Z_N = T_{N, \theta_N} \circ \cdots \circ T_{1, \theta_1} (Z_0) , \quad Z_0 \sim q_0,\quad  Z_N \sim q_{\theta}, 
```
where $\theta = (\theta_1, \dots, \theta_N)$ are the parameters to be learned,
and $q_{\theta}$ is the transformed distribution (typically called the
variational distribution or the flow distribution). 
This describes **sampling procedure** of normalizing flows, which requires
sending draws from the base distribution through a forward pass of these flow layers.

Since all the transformations are invertible (technically diffeomorphic), we
can evaluate the density of a normalizing flow distribution $q_{\theta}$ by the
change of variable formula:
```math
q_\theta(x)=\frac{q_0\left(T_1^{-1} \circ \cdots \circ
T_N^{-1}(x)\right)}{\prod_{n=1}^N J_n\left(T_n^{-1} \circ \cdots \circ
T_N^{-1}(x)\right)} \quad J_n(x)=\left|\operatorname{det} \nabla_x
T_n(x)\right|.
```
Here we drop the subscript $\theta_n, n = 1, \dots, N$ for simplicity. 
Density evaluation of normalizing flow requires computing the **inverse** and the
**Jacobian determinant** of each flow layer.

Given the feasibility of i.i.d. sampling and density evaluation, normalizing
flows can be trained by minimizing some statistical distances to the target
distribution $p$. The typical choice of the statistical distance is the forward
and reverse Kullback-Leibler (KL) divergence, which leads to the following
optimization problems:
```math
\begin{aligned}
\text{Reverse KL:}\quad
&\argmin _{\theta} \mathbb{E}_{q_{\theta}}\left[\log q_{\theta}(Z)-\log p(Z)\right] \\
&= \argmin _{\theta} \mathbb{E}_{q_0}\left[\log \frac{q_\theta(T_N\circ \cdots \circ T_1(Z_0))}{p(T_N\circ \cdots \circ T_1(Z_0))}\right] \\
&= \argmax _{\theta} \mathbb{E}_{q_0}\left[ \log p\left(T_N \circ \cdots \circ T_1(Z_0)\right)-\log q_0(X)+\sum_{n=1}^N \log J_n\left(F_n \circ \cdots \circ F_1(X)\right)\right]
\end{aligned}
```
and 
```math
\begin{aligned}
\text{Forward KL:}\quad
&\argmin _{\theta} \mathbb{E}_{p}\left[\log q_{\theta}(Z)-\log p(Z)\right] \\
&= \argmin _{\theta} \mathbb{E}_{p}\left[\log q_\theta(Z)\right] 
\end{aligned}
```
Both problems can be solved via standard stochastic optimization algorithms,
such as stochastic gradient descent (SGD) and its variants. 



