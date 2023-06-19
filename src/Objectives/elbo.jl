using Distributions, LinearAlgebra
using Bijectors
using Random

###############
# TODO: 
###############
# 1. so far we assume reference q isa Distribution, and is compatible with Bijectors.jl so that we can construct a `flow::Bijectors.TransformedDistribution`
#       we should ensure better generality such that we only need `logq` and `T<:Bijectors.Bijector, T⁻¹` for the input. 

# 2. make types for variational objectives.
#       - ELBO 
#       - MLE
#       - IWAE
#       - f-divergence

####################################
# training by minimizing reverse KL
####################################    
function elbo_single_sample(
    flow::Bijectors.TransformedDistribution,     # variational distribution to be trained (flow = T q₀, where T <: Bijectors.Bijector, q₀ reference dist that one can easily sample and compute logpdf)
    logp,                                       # lpdf (unnormalized) of the target distribution
    x,                                          # sample from reference dist q
)
    y, logabsdetjac = with_logabsdet_jacobian(flow.transform, x)
    return logp(y) - logpdf(flow.dist, x) + logabsdetjac
end

# ELBO based on multiple iid samples
function elbo(
    flow::Bijectors.UnivariateTransformed,      # variational distribution to be trained
    logp,                                       # lpdf (unnormalized) of the target distribution
    xs::AbstractVector,                          # samples from reference dist q
)
    elbo_values = map(x -> elbo_single_sample(flow, logp, x), xs)
    return mean(elbo_values)
end

function elbo(
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    logp,                                       # lpdf (unnormalized) of the target distribution
    xs::AbstractMatrix,                         # samples from reference dist q
)
    elbo_values = map(x -> elbo_single_sample(flow, logp, x), eachcol(xs))
    return mean(elbo_values)
end

function elbo(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, n_samples)
    return elbo(flow, logp, rand(rng, flow.dist, n_samples))
end

function elbo(rng::AbstractRNG, flow::Bijectors.UnivariateTransformed, logp, n_samples)
    return elbo(flow, logp, rand(rng, flow.dist, n_samples))
end
