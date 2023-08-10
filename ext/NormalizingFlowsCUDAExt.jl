module NormalizingFlowsCUDAExt

using CUDA
using NormalizingFlows: Random, Distributions

# FIXME: These overloads are not triggered.
# TODO: Are we even allowed these sorts of overloads in extensions?

# Make allocation of output array live on GPU.
function Distributions.rand(
    rng::CUDA.RNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous}
)
    return @inbounds Distributions.rand!(rng, Distributions.sampler(s), CuArray{float(eltype(s))}(undef, size(s)))
end

function Distributions._rand!(rng::CUDA.RNG, d::Distributions.MvNormal, x::AbstractVector)
    # Replaced usage of scalar indexing.
    Random.randn!(rng, x)
    Distributions.unwhiten!(d.Σ, x)
    x .+= d.μ
    return x
end

function Distributions.insupport(::Type{D},x::CuVector{T}) where {T<:Real,D<:Distributions.AbstractMvLogNormal}
    return all(0 .< x .< Inf)
end

end
