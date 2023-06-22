module NormalizingFlows

using Bijectors
using Optimisers
using LinearAlgebra, Random, Distributions, StatsBase
using ProgressMeter
using ADTypes, DiffResults
using Zygote, ForwardDiff, ReverseDiff, Enzyme

using DocStringExtensions

export NF, elbo, loglikelihood

"""
    NF(rng::AbstractRNG, vo, flow, args...; kwargs...)
    NF(vo, flow, args...; kwargs...) = NF(Random.default_rng(), vo, flow, args...; kwargs...)

Train the given normalizing flow `flow` by calling `train`.

# Arguments
- `rng::AbstractRNG`: random number generator
- `vo`: variational objective
- `flow`: normalizing flow to be trained
- `args...`: additional arguments for `vo`
- `max_iters::Int=1000`: maximum number of iterations
- `optimiser::Optimisers.AbstractRule=Optimisers.ADAM()`: optimiser to compute the steps
- `ADbackend::ADTypes.AbstractADType=ADTypes.AutoZygote()`: automatic differentiation backend
- `kwargs...`: additional keyword arguments for `train` (See `train`)

# Returns
- `flow_trained`: trained normalizing flow
- `opt_stats`: statistics of the optimiser during the training process (See `train`)
- `st`: optimiser state for potential continuation of training
"""
function NF end

NF(vo, flow, args...; kwargs...) = NF(Random.default_rng(), vo, flow, args...; kwargs...)
function NF(
    rng::AbstractRNG,
    vo,                                      # elbo, likelihood, f-div, STL, etc.. (how do we deal with this? it would require different input)
    flow,                                    # normalizing flow to be trained
    args...;                                 # additional arguments for vo 
    max_iters::Int=1000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    ADbackend::ADTypes.AbstractADType=ADTypes.AutoZygote(),
    kwargs...,
)
    # destruct flow for explicit access to the parameters
    @info "destructuring flow object"
    θ_flat, re = Optimisers.destructure(flow)

    # Normalizing flow training loop 
    θ_flat_trained, opt_stats, st = train(
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
include("Objectives/objs.jl")

end
