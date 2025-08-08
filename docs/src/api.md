# API

```@index
```

## Main Function

```@docs
NormalizingFlows.train_flow
```

The flow object can be constructed by `transformed` function in `Bijectors.jl`.
For example, for Gaussian VI, we can construct the flow as follows:

```julia
using Distributions, Bijectors
T = Float32
@leaf MvNormal # to prevent params in q₀ from being optimized
q₀ = MvNormal(zeros(T, 2), ones(T, 2))
flow = Bijectors.transformed(q₀, Bijectors.Shift(zeros(T,2)) ∘ Bijectors.Scale(ones(T, 2)))
```

To train the Gaussian VI targeting distribution `p` via ELBO maximization, run:

```julia
using NormalizingFlows, Optimisers

sample_per_iter = 10
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters = 2_000,
    optimiser = Optimisers.ADAM(0.01 * one(T)),
)
```

## Coupling-based flows (default constructors)

These helpers construct commonly used coupling-based flows with sensible defaults.

```@docs
NormalizingFlows.realnvp
NormalizingFlows.nsf
NormalizingFlows.RealNVP_layer
NormalizingFlows.NSF_layer
NormalizingFlows.AffineCoupling
NormalizingFlows.NeuralSplineCoupling
NormalizingFlows.create_flow
```

## Variational Objectives

We provide ELBO (reverse KL) and expected log-likelihood (forward KL). You can also
supply your own objective with the signature `vo(rng, flow, args...)`.

### Evidence Lower Bound (ELBO)

By maximizing the ELBO, it is equivalent to minimizing the reverse KL divergence between $q_\theta$ and $p$:

```math
\begin{aligned}
&\min _{\theta} \mathbb{E}_{q_{\theta}}\left[\log q_{\theta}(Z)-\log p(Z)\right]  \quad \text{(Reverse KL)}\\
& = \max _{\theta} \mathbb{E}_{q_0}\left[ \log p\left(T_N \circ \cdots \circ
T_1(Z_0)\right)-\log q_0(X)+\sum_{n=1}^N \log J_n\left(F_n \circ \cdots \circ
F_1(X)\right)\right] \quad \text{(ELBO)}
\end{aligned}
```

Reverse KL minimization is typically used for Bayesian computation when only `logp` is available.

```@docs
NormalizingFlows.elbo
```

```@docs
NormalizingFlows.elbo_batch
```

### Log-likelihood

By maximizing the log-likelihood, it is equivalent to minimizing the forward KL divergence between $q_\theta$ and $p$:

```math
\begin{aligned}
& \min_{\theta} \mathbb{E}_{p}\left[\log q_{\theta}(Z)-\log p(Z)\right] \quad \text{(Forward KL)} \\
& = \max_{\theta} \mathbb{E}_{p}\left[\log q_{\theta}(Z)\right] \quad \text{(Expected log-likelihood)}
\end{aligned}
```

Forward KL minimization is typically used for generative modeling when samples from `p` are given.

```@docs
NormalizingFlows.loglikelihood
```

## Training Loop

```@docs
NormalizingFlows.optimize
```
