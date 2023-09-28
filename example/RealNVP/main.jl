using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
using Flux: f32
using Plots
include("../common.jl")
include("AffineCoupling.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../targets/banana.jl")

# create target p
p = Banana(2, 1.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

######################################
# learn the target using Affine coupling flow
######################################
d = 2
hdims = 2
Ls = [AffineCoupling(d, hdims, [1]) ∘ AffineCoupling(d, hdims, [2]) for i in 1:5]
ts = ∘(Ls...)
q0 = MvNormal(zeros(Float32, 2), I)
flow = transformed(q0, ts)
flow_untrained = deepcopy(flow)

# train the flow
sample_per_iter = 10
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
flow_trained, stats, _ = train_flow(
    elbo_batch,
    flow,
    logp,
    sample_per_iter;
    max_iters=200_00,
    optimiser=Optimisers.Adam(),
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses = map(x -> x.loss, stats)

######################################
# evaluate trained flow (one can see that the performance is better than the plannar flow)
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, p, 1000)

LsBN = [AffineCouplingBN(d, hdims, [1]) ∘ AffineCouplingBN(d, hdims, [2]) for i in 1:5]
tsBN = ∘(LsBN...)
q0 = MvNormal(zeros(Float32, 2), I)
flowBN = transformed(q0, tsBN)
flow_untrained = deepcopy(flowBN)

# train the flow
sample_per_iter = 10
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
flow_trained, stats, _ = train_flow(
    elbo_batch,
    flowBN,
    logp,
    sample_per_iter;
    max_iters=200_00,
    optimiser=Optimisers.Adam(),
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses = map(x -> x.loss, stats)

######################################
# evaluate trained flow (one can see that the performance is better than the plannar flow)
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, p, 1000)
