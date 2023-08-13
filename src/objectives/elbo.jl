####################################
# training by minimizing reverse KL
####################################    
function elbo_single_sample(flow::Bijectors.TransformedDistribution, logp, x)
    y, logabsdetjac = with_logabsdet_jacobian(flow.transform, x)
    return logp(y) - logpdf(flow.dist, x) + logabsdetjac
end

"""
    elbo(flow, logp, xs) 
    elbo([rng, ]flow, logp, n_samples)

Compute the ELBO for a batch of samples `xs` from the reference distribution `flow.dist`.
By maximizing the ELBO, it is equivalent to minimizing
the reverse KL divergence between ``q_\\theta`` and p, i.e., 
```math 
\begin{aligned}
&\min _{\theta} \mathbb{E}_{q_{\theta}}\left[\log q_{\theta}(Z)-\log p(Z)\right]  \quad \text{reverse KL}\\
&= \max _{\theta} \mathbb{E}_{q_0}\left[ \log p\left(T_N \circ \cdots \circ
T_1(Z_0)\right)-\log q_0(X)+\sum_{n=1}^N \log J_n\left(F_n \circ \cdots \circ
F_1(X)\right)\right] \quad \text{ELBO} 
\end{aligned}
```

# Arguments
- `rng`: random number generator
- `flow`: variational distribution to be trained. In particular 
  "flow = transformed(q₀, T::Bijectors.Bijector)", 
  q₀ is a reference distribution that one can easily sample and compute logpdf
- `logp`: log-pdf of the target distribution (not necessarily normalized)
- `xs`: samples from reference dist q₀
- `n_samples`: number of samples from reference dist q₀
"""
# ELBO based on multiple iid samples
function elbo(flow::Bijectors.UnivariateTransformed, logp, xs::AbstractVector)
    elbo_values = map(x -> elbo_single_sample(flow, logp, x), xs)
    return mean(elbo_values)
end

function elbo(flow::Bijectors.MultivariateTransformed, logp, xs::AbstractMatrix)
    elbo_values = map(x -> elbo_single_sample(flow, logp, x), eachcol(xs))
    return mean(elbo_values)
end

function elbo(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, n_samples)
    return elbo(flow, logp, rand(rng, flow.dist, n_samples))
end

function elbo(rng::AbstractRNG, flow::Bijectors.UnivariateTransformed, logp, n_samples)
    return elbo(flow, logp, rand(rng, flow.dist, n_samples))
end

function elbo(flow::Bijectors.TransformedDistribution, logp, n_samples)
    return elbo(Random.default_rng(), flow, logp, n_samples)
end