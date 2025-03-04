####################################
# training by minimizing forward KL (MLE)
####################################    
"""
    loglikelihood(rng, flow::Bijectors.TransformedDistribution, xs::AbstractVecOrMat)

Compute the log-likelihood for variational distribution flow at a batch of samples xs from 
the target distribution p. 

# Arguments
- `rng`: random number generator (empty argument, only needed to ensure the same signature as other variational objectives)
- `flow`: variational distribution to be trained. In particular 
  "flow = transformed(q₀, T::Bijectors.Bijector)", 
  q₀ is a reference distribution that one can easily sample and compute logpdf
- `xs`: samples from the target distribution p.

"""
function loglikelihood(
    ::AbstractRNG,                          # empty argument
    flow::Bijectors.UnivariateTransformed,    # variational distribution to be trained
    xs::AbstractVector,                       # sample batch from target dist p
)
    return mean(Base.Fix1(logpdf, flow), xs)
end

function loglikelihood(
    ::AbstractRNG,                           # empty argument
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    xs::AbstractMatrix,                         # sample batch from target dist p
)
    llhs = map(x -> logpdf(flow, x), eachcol(xs))
    return mean(llhs)
end

## TODO:will need to implement the version that takes a dataloader 
# function loglikelihood(
#     rng::AbstractRNG,
#     flow::Bijectors.TransformedDistribution,
#     dataloader
# )
#     xs = dataloader(rng)
#     return loglikelihood(rng, flow, collect(dataloader))
# end
