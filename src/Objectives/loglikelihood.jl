using Distributions, LinearAlgebra
using Bijectors
using Random

####################################
# training by minimizing forward KL (MLE)
####################################    
function llh_single_sample(
    flow::Bijectors.TransformedDistribution,     # variational distribution to be trained
    logq,                                       # lpdf (exact) of the reference distribution
    x,                                          # sample from target dist p
)
    b = inverse(flow.transform)
    y, logjac = with_logabsdet_jacobian(b, x)
    return logq(y) + logjac
end

function loglikelihood(
    flow::Bijectors.UnivariateTransformed,    # variational distribution to be trained
    logq,                                     # lpdf (exact) of the reference distribution
    xs::AbstractVector,                       # sample from target dist p
)
    llhs = map(x -> llh_single_sample(flow, logq, x), xs)
    return mean(llhs)
end

function loglikelihood(
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    logq,                                        # lpdf (exact) of the reference distribution
    xs::AbstractMatrix,                         # sample from target dist p
)
    llhs = map(x -> llh_single_sample(flow, logq, x), eachcol(xs))
    return mean(llhs)
end