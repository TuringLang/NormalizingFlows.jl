# this file defines rand_device function to sample from a Distribution or a
# flow<:Bijectors.TranformedDistribution 
# this is mainly for resolving the issue of sampling from a distribution on GPU 

# function rand_device(
#     rng::AbstractRNG,
#     s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
# )
#     if !(rng isa CUDA.RNG)
#         return Distributions.rand(rng, s)
#     else
#         return rand_cuda(rng, s)
#     end
# end

function rand_device(
    rng::AbstractRNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return Distributions.rand(rng, s)
end

function rand_device(
    rng::AbstractRNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    return Distributions.rand(rng, s, n)
end

# function rand_device(
#     rng::AbstractRNG,
#     s::Distributions.Sampleable{Distributions.Multivariate,Distributions.Continuous},
#     n::Int,
# )
#     if !(rng isa CUDA.RNG)
#         return Distributions.rand(rng, s, n)
#     else
#         return rand_cuda(rng, s, n)
#     end
# end

#########################
# for Bijectors.jl
##########################

function rand_device(rng::AbstractRNG, td::Bijectors.TransformedDistribution)
    return Distributions.rand(rng, td)
end

function rand_device(rng::AbstractRNG, td::Bijectors.TransformedDistribution, n::Int)
    return Distributions.rand(rng, td, n)
end

# function rand_device(rng::AbstractRNG, td::Bijectors.TransformedDistribution)
#     if !(rng isa CUDA.RNG)
#         return Distributions.rand(rng, td)
#     else
#         return rand_cuda(rng, td)
#     end
# end

# function rand_device(rng::AbstractRNG, td::Bijectors.TransformedDistribution, n::Int)
#     if !(rng isa CUDA.RNG)
#         return Distributions.rand(rng, td, n)
#     else
#         return rand_cuda(rng, td, n)
#     end
# end
