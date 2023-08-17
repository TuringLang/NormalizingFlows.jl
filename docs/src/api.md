## API

```@index
```


## Main Function

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
## Variational Objectives
We have implemented two variational objectives, namely, ELBO and the log-likelihood objective. 
Users can also define their own objective functions, and pass it to the [`train_flow`](@ref) function.
`train_flow` will optimize the flow parameters by maximizing `vo`.
The objective function should take the following general form:
```julia
vo(rng, flow, args...) 
```
where `rng` is the random number generator, `flow` is the flow object, and `args...` are the
additional arguments that users can pass to the objective function.

#### Evidence Lower Bound (ELBO)
By maximizing the ELBO, it is equivalent to minimizing
the reverse KL divergence between $q_\theta$ and $p$, i.e., 
```math 
\begin{aligned}
&\min _{\theta} \mathbb{E}_{q_{\theta}}\left[\log q_{\theta}(Z)-\log p(Z)\right]  \quad \text{(Reverse KL)}\\
& = \max _{\theta} \mathbb{E}_{q_0}\left[ \log p\left(T_N \circ \cdots \circ
T_1(Z_0)\right)-\log q_0(X)+\sum_{n=1}^N \log J_n\left(F_n \circ \cdots \circ
F_1(X)\right)\right] \quad \text{(ELBO)} 
\end{aligned}
```
Reverse KL minimization is typically used for **Bayesian computation**, 
where one only has access to the log-(unnormalized)density of the target distribution $p$ (e.g., a Bayesian posterior distribution), 
and hope to generate approximate samples from it.

```@docs
NormalizingFlows.elbo
```
#### Log-likelihood

By maximizing the log-likelihood, it is equivalent to minimizing the forward KL divergence between $q_\theta$ and $p$, i.e., 
```math 
\begin{aligned}
& \min_{\theta} \mathbb{E}_{p}\left[\log q_{\theta}(Z)-\log p(Z)\right] \quad \text{(Forward KL)} \\
& = \max_{\theta} \mathbb{E}_{p}\left[\log q_{\theta}(Z)\right] \quad \text{(Expected log-likelihood)}
\end{aligned}
```
Forward KL minimization is typically used for **generative modeling**, 
where one is given a set of samples from the target distribution $p$ (e.g., images)
and aims to learn the density or a generative process that outputs high quality samples.

```@docs
NormalizingFlows.loglikelihood
```


## Training Loop

```@docs
NormalizingFlows.optimize
```


## Utility Functions for Taking Gradient
```@docs
NormalizingFlows.grad!
NormalizingFlows.value_and_gradient!
```

