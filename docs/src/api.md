# API

```@index
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
T_1(Z_0)\right)-\log q_0(X)+\sum_{n=1}^N \log J_n\left(T_n \circ \cdots \circ
T_1(X)\right)\right] \quad \text{(ELBO)}
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


## Available Flows

`NormalizingFlows.jl` provides two commonly used normalizing flows---`RealNVP` and
`Neural Spline Flow (NSF)`---and two simple flows---`Planar Flow` and `Radial Flow`.

### RealNVP (Affine Coupling Flow)

These helpers construct commonly used coupling-based flows with sensible defaults.

```@docs
NormalizingFlows.realnvp
NormalizingFlows.RealNVP_layer
NormalizingFlows.AffineCoupling
```

### Neural Spline Flow (NSF)

```@docs
NormalizingFlows.nsf
NormalizingFlows.NSF_layer
NormalizingFlows.NeuralSplineCoupling
```

#### Planar and Radial Flows

```@docs
NormalizingFlows.planarflow
NormalizingFlows.radialflow
```

## Utility Functions

```@docs
NormalizingFlows.create_flow
```

```@docs
NormalizingFlows.fnn
```

