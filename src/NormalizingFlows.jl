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
import DifferentiationInterface as DI

using DocStringExtensions

export train_flow, elbo, elbo_batch, loglikelihood

"""
    train_flow([rng::AbstractRNG, ]vo, flow, args...; kwargs...)

Train the given normalizing flow `flow` by calling `optimize`.

# Arguments
- `rng::AbstractRNG`: random number generator
- `vo`: variational objective
- `flow`: normalizing flow to be trained, we recommend to define flow as `<:Bijectors.TransformedDistribution` 
- `args...`: additional arguments for `vo`

# Keyword Arguments
- `max_iters::Int=1000`: maximum number of iterations
- `optimiser::Optimisers.AbstractRule=Optimisers.ADAM()`: optimiser to compute the steps
- `ADbackend::ADTypes.AbstractADType=ADTypes.AutoZygote()`: 
    automatic differentiation backend, currently supports
    `ADTypes.AutoZygote()`, `ADTypes.ForwardDiff()`, `ADTypes.ReverseDiff()`, 
    `ADTypes.AutoMooncake()` and
    `ADTypes.AutoEnzyme(;
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
        function_annotation=Enzyme.Const,
    )`.
    If user wants to use `AutoEnzyme`, please make sure to include the `set_runtime_activity` and `function_annotation` as shown above.
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


# interface of contructing common flow layers
include("flows/utils.jl")
include("flows/realnvp.jl")
include("flows/neuralspline.jl")
# a new implementation of Neural Spline Flow based on MonotonicSplines.jl
# the construction of the RQS seems to be more efficient than the one in Bijectors.jl
# and supports batched operations.
include("flows/new_nsf.jl")

export create_flow
export AffineCoupling, RealNVP_layer, realnvp
export NeuralSplineCoupling, NSF_layer, nsf
export NSC, new_NSF_layer, new_nsf


end
