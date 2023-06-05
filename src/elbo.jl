using Distributions, LinearAlgebra
using Bijectors
using Random

####################################
# training by minimizing reverse KL
####################################    
function elbo_single_sample(
    x::AbstractVector,                          # sample from reference dist q
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    logp,                                       # lpdf (unnormalized) of the target distribution
    logq                                        # lpdf (exact) of the reference distribution
    )
    y, logabsdetjac = with_logabsdet_jacobian(flow.transform, x)
    return logp(y) - logq(x) + logabsdetjac
end

# ELBO based on multiple iid samples
function elbo(
    xs::AbstractMatrix,                         # samples from reference dist q
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    logp,                                       # lpdf (unnormalized) of the target distribution
    logq                                        # lpdf (exact) of the reference distribution
    )
    n_samples = size(xs, 2) # each column is a sample
    elbo_values = map(x -> elbo_single_sample(x, flow, logp, logq), eachcol(xs))
    return sum(elbo_values) / n_samples
end

elbo(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, logq, n_samples) = elbo(
    rand(rng, flow.dist, n_samples), flow, logp, logq
)

####################################
# training by minimizing forward KL (MLE)
####################################    