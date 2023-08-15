module NormalizingFlowsCUDAExt

using CUDA
using NormalizingFlows: Random, Distributions

# Make allocation of output array live on GPU.
function Distributions.rand(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return @inbounds Distributions.rand!(
        rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, size(s))
    )
end

function Distributions.rand(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    return @inbounds Distributions.rand!(
        rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, length(s), n)
    )
end

function Distributions._rand!(rng::CUDA.RNG, d::Distributions.MvNormal, x::CuVecOrMat)
    # Replaced usage of scalar indexing.
    CUDA.randn!(rng, x)
    Distributions.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end

function Distributions.insupport(
    ::Type{D}, x::CuVector{T}
) where {T<:Real,D<:Distributions.AbstractMvLogNormal}
    return all(0 .< x .< Inf)
end

end
