using Bijectors
using CUDA
using Distributions
using LinearAlgebra
using MacroTools
using Random
using Functors
using Flux
##
struct BatchDistributionWrapper{D<:Distribution, T<:AbstractArray}
    distribution::Type{D}
    parameters::NTuple{M, T} where M
    batch_shape::NTuple{N, Int} where N
end

@functor BatchDistributionWrapper (parameters, )

function BatchDistributionWrapper(dist::Symbol, params, batch_shape=())
    if any((iszero∘ndims), params) # if any of the parameters is a scalar, just return the Distribution
        return getfield(Distributions, dist)(params...)
    end

    @assert isdefined(Distributions, dist) "Distribution $dist is not defined"
    
    @assert all(map(x->eltype(x) != Any, params)) "all parameters should have the concrete element type"
    @assert all(map(x->eltype(x) == eltype(params[1]), params)) "all parameters should have the same element type"

    D = getfield(Distributions, dist)
    return BatchDistributionWrapper{D, eltype(params)}(D, params, batch_shape)
end

macro batch(dist_with_args, batch_shape)
    c = @capture(dist_with_args, dist_name_(args__))
    @assert c "$dist_with_args should be a distribution with arguments"
    quote
        BatchDistributionWrapper($(Meta.quot(dist_name)), (tuple($(args...))), $(batch_shape))
    end
end

d = @batch Normal(zeros(2, 2, 1), ones(2, 2, 1)) (2, 2)

# Default implementation
function Random.rand(rng::AbstractRNG, d::BatchDistributionWrapper{D, T}) where {D, T} 
    dists = D.(d.parameters...)
    return map(rand, dists)
end

function Random.rand(rng::AbstractRNG, d::BatchDistributionWrapper{D, T}, sample_shape::Tuple{Vararg{Int, N}}) where {D, T, N}
    dists = D.(d.parameters...)
    event_shape = length(rand(rng, dists[1]))
    samples = map(d -> reshape(rand(rng, d, sample_shape), (sample_shape..., 1)), dists)
    reshaped_samples = reshape(cat(samples..., dims=N+1), (sample_shape..., d.batch_shape..., event_shape))
    return reshaped_samples
end

function Random.rand(rng::AbstractRNG, d::BatchDistributionWrapper{D, T}) where {D<:Normal, T<:AbstractArray}
    μ, σ = d.parameters
    x = similar(μ)
    rand!(x)
    x .*= σ
    x .+= μ
    return x
end

function Random.rand(rng::AbstractRNG, d::BatchDistributionWrapper{D, T}) where {D<:Normal, T<:CuArray}
    μ, σ = d.parameters
    x = similar(μ)
    CUDA.rand!(x)
    x .*= σ
    x .+= μ
    return x
end

function Distributions.logpdf(d::BatchDistributionWrapper{D, T}, x::AbstractArray) where {D, T}
    dists = D.(d.parameters...)
    return reshape(map(logpdf, dists, x), d.batch_shape)
end

# both CPU and GPU
function Distributions.logpdf(d::BatchDistributionWrapper{D, T}, x::AbstractArray) where {D <: Normal, T}
    μ, σ = d.parameters
    return -0.5 * (((x .- μ) ./ σ).^2 .+ log(2π) .+ 2 .* log.(σ))
end
