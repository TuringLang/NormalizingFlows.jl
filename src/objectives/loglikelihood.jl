####################################
# training by minimizing forward KL (MLE)
####################################    
"""
    loglikelihood(flow::Bijectors.TransformedDistribution, xs::AbstractVecOrMat)

Compute the log-likelihood for variational distribution flow at a batch of samples xs from 
the target distribution.

"""
function loglikelihood(
    flow::Bijectors.UnivariateTransformed,    # variational distribution to be trained
    xs::AbstractVector,                       # sample batch from target dist p
)
    return mean(Base.Fix1(logpdf, flow), xs)
end

function loglikelihood(
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    xs::AbstractMatrix,                         # sample batch from target dist p
)
    llhs = map(x -> logpdf(flow, x), eachcol(xs))
    return mean(llhs)
end

function loglikelihood(
    rng::AbstractRNG,
    flow::Bijectors.UnivariateTransformed,
    fulldata::AbstractVector,
    batchsize::Int,
)
    N = length(fulldata)
    @assert batchsize <= N
    idx = sample(rng, 1:N, batchsize; replace=false)
    xs = @view(fulldata[idx])
    return loglikelihood(flow, xs)
end

function loglikelihood(
    rng::AbstractRNG,
    flow::Bijectors.MultivariateTransformed,
    fulldata::AbstractMatrix,
    batchsize::Int,
)
    N = size(fulldata, 2)
    @assert batchsize <= N
    idx = sample(rng, 1:N, batchsize; replace=false)
    xs = @view(fulldata[:, idx])
    return loglikelihood(flow, xs)
end