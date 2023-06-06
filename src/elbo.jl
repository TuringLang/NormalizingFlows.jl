using Distributions, LinearAlgebra
using Bijectors
using Random

####################################
# training by minimizing reverse KL
####################################    
function elbo_single_sample(
    flow::Bjectors.TransformedDistribution,     # variational distribution to be trained
    logp,                                       # lpdf (unnormalized) of the target distribution
    x                                           # sample from reference dist q
    ) 
    y, logabsdetjac = with_logabsdet_jacobian(flow.transform, x)
    return logp(y) - logpdfs(flow.dist, x) + logabsdetjac
end

# ELBO based on multiple iid samples
function elbo(
    flow::Bijectors.UnivariateTransformed,      # variational distribution to be trained
    logp,                                       # lpdf (unnormalized) of the target distribution
    xs::AbstractVector                          # samples from reference dist q
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

elbo(rng::AbstractRNG, flow::Bijectors.MultivariateTransformed, logp, n_samples) = elbo(
    flow, logp, rand(rng, flow.dist, n_samples)
)

elbo(rng::AbstractRNG, flow::Bijectors.UnivariateTransformed, logp, n_samples) = elbo(
    flow, logp, rand(rng, flow.dist, n_samples)
)
####################################
# training by minimizing forward KL (MLE)
####################################    
function neg_llh_single_sample(
    flow::Bjectors.TransformedDistribution,     # variational distribution to be trained
    logq,                                       # lpdf (exact) of the reference distribution
    x,                                          # sample from target dist p
    ) 
    b = inverse(flow.transform)
    y, logjac = with_logabsdet_jacobian(b, x)
    return -logq(y) - logjac
end
    
function neg_llh(
    flow::Bijectors.UnivariateTransformed,    # variational distribution to be trained
    logq,                                     # lpdf (exact) of the reference distribution
    xs::AbstractVector,                       # sample from target dist p
    )
    neg_llhs = map(x -> neg_llh_single_sample(flow, logq, x), xs)
    return mean(neg_llhs) 
end
   
function neg_llh(
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    logq,                                        # lpdf (exact) of the reference distribution
    xs::AbstractMatrix,                         # sample from target dist p
    )
    neg_llhs = map(x -> neg_llh_single_sample(flow, logq, x), eachcol(xs))
    return mean(neg_llhs)
end
   
