using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
using Flux: f32
using Plots

const NF = NormalizingFlows
include("../common.jl")
include("nsf_layer.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# load dataset
######################################
include("../targets/banana.jl")
p = Banana(2, 1.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

d = 2
hdims = 10
K = 8
B = 3
Ls = [
    NeuralSplineLayer(d, hdims, K, B, [1]) ∘ NeuralSplineLayer(d, hdims, K, B, [2]) for
    i in 1:2
]
q0 = MvNormal(zeros(Float32, d), I)
flow = Bijectors.transformed(q0, ∘(Ls...))
# flow = Bijectors.transformed(q0, trans)
flow_untrained = deepcopy(flow)

sample_per_iter = 10
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters=200_00,
    optimiser=Optimisers.Adam(1e-2),
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses = map(x -> x.loss, stats)

######################################
# evaluate trained flow (one can see that the performance is better than the plannar flow)
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss