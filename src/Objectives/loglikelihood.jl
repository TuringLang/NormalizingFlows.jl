####################################
# training by minimizing forward KL (MLE)
####################################    
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
    return mean(Base.Fix1(logpdf, flow), eachcol(xs))
end