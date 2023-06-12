module NormalizingFlows

using Bijectors
using Optimisers
using ProgressMeter
using LinearAlgebra
using Zygote
using DiffResults
import AbstractDifferentiation as AD

include("train.jl")
include("elbo.jl")

export NF, elbo

function NF(
    vo,                                      # elbo, likelihood, f-div, STL, etc.. (how do we deal with this? it would require different input)
    flow::Bijectors.TransformedDistribution, # flow = T q₀, where T <: Bijectors.Bijector, q₀ reference dist that one can easily sample and compute logpdf
    args...;                                 # additional arguments for vo 
    rng::AbstractRNG=Random.GLOBAL_RNG,
    max_iters::Int=1000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    AD_backend::AD.AbstractBackend=AD.ZygoteBackend(),
)
    # destruct flow for explicit access to the parameters
    # destructure can result in some overhead when the flow length is large
    @info "desctructuring flow..."
    θ_flat, re = Flux.destructure(flow)

    # Normalizing flow training loop 
    @info "start training..."
    losses, θ_flat_trained, st = train!(
        AD_backend,
        vo,
        θ_flat,
        re,
        args...;
        max_iters=max_iters,
        optimiser=optimiser,
        rng=rng,
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, losses, st
end

end
