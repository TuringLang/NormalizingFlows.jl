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

function NF(
    vo,                                      # elbo, likelihood, f-div, STL, etc.. (how do we deal with this? it would require different input)
    flow::Bijectors.TransformedDistribution, # flow = T q₀, where T <: Bijectors.Bijector, q₀ reference dist that one can easily sample and compute logpdf
    args...;                                 # additional arguments for vo 
    rng::AbstractRNG=Random.GLOBAL_RNG,
    max_iters::Int=1000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    at::ADTypes.AbstractADType=ADTypes.AutoZygote,
)
    # destruct flow for explicit access to the parameters
    # destructure can result in some overhead when the flow length is large
    @info "desctructuring flow..."
    θ_flat, re = Optimisers.destructure(flow)

    # Normalizing flow training loop 
    @info "start training..."
    losses, θ_flat_trained, st = train!(
        at, vo, θ_flat, re, args...; max_iters=max_iters, optimiser=optimiser, rng=rng
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, losses, st
end

end
