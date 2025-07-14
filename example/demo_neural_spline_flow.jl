using Flux
using Bijectors
using Bijectors: partition, combine, PartitionMask

using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Mooncake
using NormalizingFlows

include("SyntheticTargets.jl")
include("utils.jl")

##################################
# start demo
#################################
Random.seed!(123)
rng = Random.default_rng()
T = Float32

######################################
# neals funnel target
######################################
target = Funnel(2, 0.0f0, 9.0f0)
logp = Base.Fix1(logpdf, target)

######################################
# learn the target using Affine coupling flow
######################################
@leaf MvNormal
q0 = MvNormal(zeros(T, 2), ones(T, 2))

d = 2
hdims = 64
K = 10
B = 30
Ls = [
    NeuralSplineLayer(d, hdims, K, B, [1]) ∘ NeuralSplineLayer(d, hdims, K, B, [2]) for
    i in 1:3
]

flow = create_flow(Ls, q0)
flow_untrained = deepcopy(flow)


######################################
# start training
######################################
sample_per_iter = 64

# callback function to log training progress
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
# TODO: now using AutoMooncake the example broke, but AutoZygote works, need to debug
adtype = ADTypes.AutoMooncake(; config = Mooncake.Config())
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters=100,   # change to larger number of iterations (e.g., 50_000) for better results
    optimiser=Optimisers.Adam(5e-5),
    ADbackend=adtype,
    show_progress=true,
    callback=cb,
    hasconverged=checkconv,
)
θ, re = Optimisers.destructure(flow_trained)
losses = map(x -> x.loss, stats)

######################################
# evaluate trained flow
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, target, 1000)
