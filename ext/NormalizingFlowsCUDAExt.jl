module NormalizingFlowsCUDAExt

using CUDA
using NormalizingFlows
using NormalizingFlows: Bijectors, Distributions, Random
using ChainRulesCore: @non_differentiable

function NormalizingFlows._device_specific_rand(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return _cuda_rand(rng, s)
end

function NormalizingFlows._device_specific_rand(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    return _cuda_rand(rng, s, n)
end

function _cuda_rand(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return _cuda_draw(rng, s, size(s))
end

function _cuda_rand(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    return _cuda_draw(rng, s, (size(s)..., n))
end

function _cuda_draw(rng::CUDA.RNG, s, dims::Tuple)
    return @inbounds Distributions.rand!(
        rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, dims)
    )
end

# Zygote cannot trace a `CuArray` allocation: it descends into the CUDA allocator and fails
# to compile. Nothing is lost by hiding the draw, since `rand!` carries no gradient on the
# host path either. Only the draw is opaque; a flow's transform stays differentiable.
@non_differentiable _cuda_draw(::Any, ::Any, ::Any)

# ! this is type piracy
# replacing original function with scalar indexing
function Distributions._rand!(rng::CUDA.RNG, d::Distributions.MvNormal, x::CuVecOrMat)
    Random.randn!(rng, x)
    Distributions.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end

# to enable `_device_specific_rand(rng:CUDA.RNG, flow[, num_samples])`
function NormalizingFlows._device_specific_rand(rng::CUDA.RNG, td::Bijectors.TransformedDistribution)
    return _cuda_rand(rng, td)
end

function NormalizingFlows._device_specific_rand(
    rng::CUDA.RNG, td::Bijectors.TransformedDistribution, num_samples::Int
)
    return _cuda_rand(rng, td, num_samples)
end

function _cuda_rand(rng::CUDA.RNG, td::Bijectors.TransformedDistribution)
    return td.transform(_cuda_rand(rng, td.dist))
end

function _cuda_rand(rng::CUDA.RNG, td::Bijectors.TransformedDistribution, num_samples::Int)
    samples = _cuda_rand(rng, td.dist, num_samples)
    res = reduce(
        hcat,
        map(axes(samples, 2)) do i
            return td.transform(view(samples, :, i))
        end,
    )
    return res
end

end
