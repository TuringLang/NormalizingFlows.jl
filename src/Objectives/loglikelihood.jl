####################################
# training by minimizing forward KL (MLE)
####################################    
function loglikelihood(
    flow::Bijectors.UnivariateTransformed,    # variational distribution to be trained
    xs::AbstractVector,                       # sample batch from target dist p
)
    llhs = map(x -> logpdf(flow, x), xs)
    return mean(llhs)
end

function loglikelihood(
    flow::Bijectors.MultivariateTransformed,    # variational distribution to be trained
    xs::AbstractMatrix,                         # sample batch from target dist p
)
    llhs = map(x -> logpdf(flow, x), eachcol(xs))
    return mean(llhs)
end