## API

```@index
```


## Main function

```@docs
NormalizingFlows.train_flow
```

The flow object can be constructed by `transformed` function in `Bijectors.jl` package.
For example of Gaussian VI, we can construct the flow as follows:
```@julia
using Distributions, Bijectors
T= Float32
q₀ = MvNormal(zeros(T, 2), ones(T, 2))
flow = Bijectors.transformed(q₀, Bijectors.Shift(zeros(2)) ∘ Bijectors.Scale(ones(T, 2)))
```
To train the Gaussian VI targeting at distirbution $p$ via ELBO maiximization, we can run
```@julia
using NormalizingFlows

sample_per_iter = 10
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters=2_000,
    optimiser=Optimisers.ADAM(0.01 * one(T)),
)
```
## Objective functions

### ELBO

By maximizing the ELBO, it is equivalent to minimizing
the reverse KL divergence between $q_\theta$ and p, i.e., 
```math 
\begin{aligned}
&\min _{\theta} \mathbb{E}_{q_{\theta}}\left[\log q_{\theta}(Z)-\log p(Z)\right]  \quad \text{(Reverse KL)}\\
&= \max _{\theta} \mathbb{E}_{q_0}\left[ \log p\left(T_N \circ \cdots \circ
T_1(Z_0)\right)-\log q_0(X)+\sum_{n=1}^N \log J_n\left(F_n \circ \cdots \circ
F_1(X)\right)\right] \quad \text{(ELBO)} 
\end{aligned}
```
```@docs
NormalizingFlows.elbo
```

### Log-likelihood

By maximizing the log-likelihood, it is equivalent to minimizing
the forward KL divergence between ``q_\\theta`` and p, i.e., 
```math 
\begin{aligned}
& \min_{\theta} \mathbb{E}_{p}\left[\log q_{\theta}(Z)-\log p(Z)\right] \quad \text{(Forward KL)} \\
& \max_{\theta} \mathbb{E}_{p}\left[\log q_{\theta}(Z)\right] \quad \text{(Expected log-likelihood)}
\end{aligned}
```
```@docs
NormalizingFlows.loglikelihood
```


## Utilitty function for training

```@docs
NormalizingFlows.optimize
```


```@docs
NormalizingFlows.grad!
NormalizingFlows.value_and_gradient!
```

