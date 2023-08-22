module NormalizingFlowsCUDAExt

using CUDA
using NormalizingFlows
using NormalizingFlows: Random, Distributions, Bijectors

# to enable `rand_device(rng:CUDA.RNG, dist[, num_samples])`
function NormalizingFlows.rand_device(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    println("gpu rand")
    return rand_cuda(rng, s)
end

function NormalizingFlows.rand_device(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    println("gpu rand")
    return rand_cuda(rng, s, n)
end

function rand_cuda(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return @inbounds Distributions.rand!(
        rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, size(s))
    )
end

function rand_cuda(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    return @inbounds Distributions.rand!(
        rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, size(s)..., n)
    )
end

# Question: is this type piracy okay? 
# (it's probably not ideal but this is sensible enough for now )
function Distributions._rand!(rng::CUDA.RNG, d::Distributions.MvNormal, x::CuVecOrMat)
    # Replaced usage of scalar indexing.
    Random.randn!(rng, x)
    Distributions.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end

# to enable `rand_device(rng:CUDA.RNG, flow[, num_samples])`
function NormalizingFlows.rand_device(rng::CUDA.RNG, td::Bijectors.TransformedDistribution)
    return rand_cuda(rng, td)
end

function NormalizingFlows.rand_device(
    rng::CUDA.RNG, td::Bijectors.TransformedDistribution, num_samples::Int
)
    return rand_cuda(rng, td, num_samples)
end

function rand_cuda(rng::CUDA.RNG, td::Bijectors.TransformedDistribution)
    return td.transform(rand_cuda(rng, td.dist))
end

function rand_cuda(rng::CUDA.RNG, td::Bijectors.TransformedDistribution, num_samples::Int)
    samples = rand_cuda(rng, td.dist, num_samples)
    res = reduce(
        hcat,
        map(axes(samples, 2)) do i
            return td.transform(view(samples, :, i))
        end,
    )
    return res
end

end
