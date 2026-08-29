module NormalizingFlows

using ADTypes
using Distributions
using LinearAlgebra
using Optimisers
using ProgressMeter
using Random
using StatsBase
using Bijectors
using Bijectors: PartitionMask, Inverse, combine, partition
using Functors
using AbstractPPL: AbstractPPL
using LogExpFunctions: LogExpFunctions
using ChainRulesCore: @non_differentiable, ignore_derivatives
using GPUArraysCore: AbstractGPUMatrix
using PDMats: PDMat, whiten

using DocStringExtensions

export train_flow, elbo, elbo_batch, loglikelihood

"""
    train_flow([rng::AbstractRNG, ]vo, flow, args...; kwargs...)

Train the given normalizing flow `flow` by calling `optimize`.

Arguments
- `rng::AbstractRNG`: random number generator (default: `Random.default_rng()`)
- `vo`: variational objective with signature `vo(rng, flow, args...)`. 
    We implement [`elbo`](@ref), [`elbo_batch`](@ref), and [`loglikelihood`](@ref).
- `flow`: the normalizing flow---a `Bijectors.TransformedDistribution` (recommended).
    Mark the base distribution as a leaf first (`Functors.@leaf MvNormal`), otherwise
    `Optimisers.destructure` tries to flatten its covariance factorisation and fails.
- `args...`: additional arguments passed to `vo`

# Keyword Arguments
- `max_iters::Int=1000`: maximum number of iterations
- `optimiser::Optimisers.AbstractRule=Optimisers.ADAM()`: optimiser to compute the steps
- `ADbackend::ADTypes.AbstractADType`: automatic differentiation backend. Required, it has
    no default. Currently supports
    `ADTypes.AutoZygote()`, `ADTypes.AutoForwardDiff()`, `ADTypes.AutoReverseDiff()`,
    `ADTypes.AutoMooncake()` and
    `ADTypes.AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )`.
    If user wants to use `AutoEnzyme`, please make sure to include the `set_runtime_activity` and `function_annotation` as shown above.
    Gradients are computed through AbstractPPL's evaluator interface, so the chosen backend package must be loaded first.
    `AutoForwardDiff` needs `using ForwardDiff`, and `AutoMooncake` needs `using Mooncake`.
    The other backends (`AutoZygote`, `AutoReverseDiff`, `AutoEnzyme`) additionally need `using DifferentiationInterface` alongside the backend package.
    `AutoReverseDiff(; compile=true)` is rejected: a compiled tape bakes the objective's
    context into itself, so every iteration would differentiate against the first
    iteration's random draws.
- `kwargs...`: additional keyword arguments for `optimize` (See [`optimize`](@ref) for details)

# Returns
- `flow_trained`: trained normalizing flow
- `opt_stats`: statistics of the optimiser during the training process 
    (See [`optimize`](@ref) for details)
- `st`: optimiser state for potential continuation of training
"""
function train_flow(vo, flow, args...; kwargs...)
    return train_flow(Random.default_rng(), vo, flow, args...; kwargs...)
end
function train_flow(
    rng::AbstractRNG,
    vo,
    flow,
    args...;
    max_iters::Int=1000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    ADbackend::ADTypes.AbstractADType,
    kwargs...,
)
    # destruct flow for explicit access to the parameters
    # use FunctionChains instead of simple compositions to construct the flow when many flow layers are involved
    # otherwise the compilation time for destructure will be too long
    θ_flat, re = Optimisers.destructure(flow)

    loss(θ, rng, args...) = -vo(rng, re(θ), args...)

    # Normalizing flow training loop 
    θ_flat_trained, opt_stats, st = optimize(
        ADbackend,
        loss,
        θ_flat,
        re,
        rng,
        args...;
        max_iters=max_iters,
        optimiser=optimiser,
        kwargs...,
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, opt_stats, st
end

include("optimize.jl")

# objectives
include("objectives/elbo.jl")
include("objectives/loglikelihood.jl") # not fully tested

"""
    _device_specific_rand

By default dispatch to `Random.rand`, but maybe overload when the random number 
generator is device specific (e.g. `CUDA.RNG`).
"""
function _device_specific_rand end

function _device_specific_rand(
    rng::Random.AbstractRNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
)
    return Random.rand(rng, s)
end

function _device_specific_rand(
    rng::Random.AbstractRNG,
    s::Distributions.Sampleable{<:Distributions.ArrayLikeVariate,Distributions.Continuous},
    n::Int,
)
    return Random.rand(rng, s, n)
end

function _device_specific_rand(
    rng::Random.AbstractRNG, td::Bijectors.TransformedDistribution
)
    return Random.rand(rng, td)
end

function _device_specific_rand(
    rng::Random.AbstractRNG, td::Bijectors.TransformedDistribution, n::Int
)
    return Random.rand(rng, td, n)
end

"""
    _device_draw(rng, s, dims)

Draw a `dims`-shaped sample from `s` into an array on the device `rng` targets.
Device extensions add the methods.
"""
function _device_draw end

# No AD backend can trace a device allocation.
@non_differentiable _device_draw(::Any, ::Any, ::Any)

"""
    _device_specific_logpdf(d, xs)

Log-density of `d` at each column of `xs`, left on the device holding `xs`.
`Distributions.logpdf` maps over the columns and materialises a host array, so the ELBO cannot be assembled from it when the samples live on a GPU.
"""
_device_specific_logpdf(d, xs::AbstractMatrix) = logpdf(d, xs)

function _device_specific_logpdf(d::Distributions.MvNormal, xs::AbstractGPUMatrix)
    return _batched_mvnormal_logpdf(d, xs)
end

# `logdet(::Cholesky)` reads `factors[i, i]` in a host loop, which a GPU array rejects.
_cov_logdet(Σ) = logdet(Σ)
_cov_logdet(Σ::PDMat) = 2 * sum(log, diag(cholesky(Σ).factors))

# `whiten` stays on the device and does not mutate, unlike the `sqmahal` behind `logpdf`.
# `d` is held constant because differentiating two uses of a full covariance leaves a
# cotangent per use, and summing those indexes a device array element by element.
function _batched_mvnormal_logpdf(d::Distributions.MvNormal, xs::AbstractMatrix)
    T = eltype(xs)
    μ = ignore_derivatives(d.μ)
    Σ = ignore_derivatives(d.Σ)
    c = ignore_derivatives(T(length(d) * log(2 * π)) + _cov_logdet(Σ))
    # Mooncake has no GPU rule for `sum(f, x; dims)`.
    q = sum(abs2.(whiten(Σ, xs .- μ)); dims=1)
    return vec(-(c .+ q) ./ 2)
end

# interface of contructing common flow layers
include("flows/utils.jl")
include("flows/planar_radial.jl")
include("flows/realnvp.jl")
include("flows/rqs.jl")
include("flows/neuralspline.jl")

export create_flow
export planarflow, radialflow
export AffineCoupling, RealNVP_layer, realnvp
export NeuralSplineCoupling, NSF_layer, nsf

end
