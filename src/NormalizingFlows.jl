module NormalizingFlows

using Bijectors
using Optimisers
using ProgressMeter
using LinearAlgebra
using DiffResults
using ADTypes

include("train.jl")
include("Objectives/elbo.jl")

export NF, elbo

NF(vo, flow, args...; kwargs...) = NF(Random.default_rng(), vo, flow, args...; kwargs...)
function NF(
    rng::AbstractRNG,
    vo,                                      # elbo, likelihood, f-div, STL, etc.. (how do we deal with this? it would require different input)
    flow,                                    # normalizing flow to be trained
    args...;                                 # additional arguments for vo 
    max_iters::Int=1000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    ADbackend::ADTypes.AbstractADType=ADTypes.AutoZygote(),
)
    # destruct flow for explicit access to the parameters
    # destructure can result in some overhead when the flow length is large
    @info "destructuring flow..."
    θ_flat, re = Optimisers.destructure(flow)

    # Normalizing flow training loop 
    @info "start training..."
    losses, θ_flat_trained, st = train!(
        rng, ADbackend, vo, θ_flat, re, args...; max_iters=max_iters, optimiser=optimiser
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, losses, st
end

end
