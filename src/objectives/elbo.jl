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

# Arguments
- `rng`: random number generator
- `flow`: variational distribution to be trained. In particular 
  `flow = transformed(q₀, T::Bijectors.Bijector)`, 
  q₀ is a reference distribution that one can easily sample and compute logpdf
- `logp`: log-pdf of the target distribution (not necessarily normalized)
- `xs`: samples from reference dist q₀
- `n_samples`: number of samples from reference dist q₀

"""
function elbo(flow::Bijectors.UnivariateTransformed, logp, xs::AbstractVector)
    elbo_values = map(x -> elbo_single_sample(flow, logp, x), xs)
    return mean(elbo_values)
end

function elbo(flow::Bijectors.MultivariateTransformed, logp, xs::AbstractMatrix)
    elbo_values = map(x -> elbo_single_sample(flow, logp, x), eachcol(xs))
    return mean(elbo_values)
end

function elbo(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, n_samples)
    return elbo(flow, logp, _device_specific_rand(rng, flow.dist, n_samples))
end

function elbo(rng::AbstractRNG, flow::Bijectors.UnivariateTransformed, logp, n_samples)
    return elbo(flow, logp, _device_specific_rand(rng, flow.dist, n_samples))
end

function elbo(flow::Bijectors.TransformedDistribution, logp, n_samples)
    return elbo(Random.default_rng(), flow, logp, n_samples)
end


"""
    elbo_batch(flow, logp, xs) 
    elbo_batch([rng, ]flow, logp, n_samples)

Instead of broadcasting over elbo_single_sample, this function directly
computes the ELBO in a batched manner, which requires the flow.transform to be able to 
handle batched transformation directly.

This will be more efficient than `elbo` for invertible neural networks such as RealNVP, 
Neural Spline Flow, etc.

# Arguments
- `rng`: random number generator
- `flow`: variational distribution to be trained. In particular 
  `flow = transformed(q₀, T::Bijectors.Bijector)`, 
  q₀ is a reference distribution that one can easily sample and compute logpdf
- `logp`: log-pdf of the target distribution (not necessarily normalized)
- `xs`: samples from reference dist q₀
- `n_samples`: number of samples from reference dist q₀

"""
function elbo_batch(flow::Bijectors.MultivariateTransformed, logp, xs::AbstractMatrix)
    # requires the flow transformation to be able to handle batched inputs
    ys, logabsdetjac = with_logabsdet_jacobian(flow.transform, xs) 
    elbos = logp(ys) .- logpdf(flow.dist, xs) .+ logabsdetjac
    return elbos
end
function elbo_batch(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, n_samples)
    xs = _device_specific_rand(rng, flow.dist, n_samples)
    elbos = elbo_batch(flow, logp, xs)
    return mean(elbos)
end
elbo_batch(flow::Bijectors.UnivariateTransformed, logp, n_samples) = 
    elbo_batch(Random.default_rng(), flow, logp, n_samples)
