module NormalizingFlows

using Bijectors
using Optimisers
using LinearAlgebra, Random, Distributions, StatsBase
using ProgressMeter
using ADTypes, DiffResults

using DocStringExtensions

export train_flow, elbo, loglikelihood, value_and_gradient!

using ADTypes
using DiffResults

"""
    train_flow([rng::AbstractRNG, ]vo, flow, args...; kwargs...)

Train the given normalizing flow `flow` by calling `optimize`.

# Arguments
- `rng::AbstractRNG`: random number generator
- `vo`: variational objective
- `flow`: normalizing flow to be trained
- `args...`: additional arguments for `vo`


# Keyword Arguments
- `max_iters::Int=1000`: maximum number of iterations
- `optimiser::Optimisers.AbstractRule=Optimisers.ADAM()`: optimiser to compute the steps
- `ADbackend::ADTypes.AbstractADType=ADTypes.AutoZygote()`: automatic differentiation backend
- `kwargs...`: additional keyword arguments for `optimize` (See `optimize`)

# Returns
- `flow_trained`: trained normalizing flow
- `opt_stats`: statistics of the optimiser during the training process (See `optimize`)
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

    # Normalizing flow training loop 
    θ_flat_trained, opt_stats, st = optimize(
        rng,
        ADbackend,
        vo,
        θ_flat,
        re,
        args...;
        max_iters=max_iters,
        optimiser=optimiser,
        kwargs...,
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, opt_stats, st
end

include("train.jl")
include("objectives.jl")

# optional dependencies 
if !isdefined(Base, :get_extension) # check whether :get_extension is defined in Base
    using Requires
end

# Question: should Exts be loaded here or in train.jl? 
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210" include(
            "../ext/NormalizingFlowsForwardDiffExt.jl"
        )
        @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" include(
            "../ext/NormalizingFlowsReverseDiffExt.jl"
        )
        @require Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9" include(
            "../ext/NormalizingFlowsEnzymeExt.jl"
        )
        @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" include(
            "../ext/NormalizingFlowsZygoteExt.jl"
        )
        @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include(
            "../ext/NormalizingFlowsCUDAExt.jl"
        )
    end
end
end
