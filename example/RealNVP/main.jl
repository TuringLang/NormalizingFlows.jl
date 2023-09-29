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
p = Banana(4, 1.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

######################################
# learn the target using Affine coupling flow
######################################
d = 4
hdims = 4
Ls = [AffineCoupling(d, hdims, 1:2) ∘ AffineCoupling(d, hdims, 3:4) for i in 1:3]
ts = ∘(Ls...)
q0 = MvNormal(zeros(Float32, d), I)
flow = transformed(q0, ts)
flow_untrained = deepcopy(flow)

# train the flow
sample_per_iter = 20
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
compare_trained_and_untrained_flow_BN(flow_trained, flow_untrained, p, 1000)

xs = randn(Float32, 4, 10)
ys = flow_trained.transform(xs)
xs0 = inverse(flow_trained.transform)(ys)
xs0 .- xs

#######3 compare to the one with BN
LsBN = [AffineCouplingBN(d, hdims, 1:2) ∘ AffineCouplingBN(d, hdims, 3:4) for i in 1:5]
tsBN = ∘(LsBN...)
q0 = MvNormal(zeros(Float32, d), I)
flowBN = transformed(q0, tsBN)
flow_untrained_BN = deepcopy(flowBN)

# train the flow
sample_per_iter = 20
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 2e-3
flow_trained_BN, statsBN, _ = train_flow(
    elbo_batch,
    flowBN,
    logp,
    sample_per_iter;
    max_iters=300_00,
    optimiser=Optimisers.Adam(),
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses_BN = map(x -> x.loss, statsBN)

######################################
# evaluate trained flow (one can see that the performance is better than the plannar flow)
######################################
plot(losses_BN; label="Loss", linewidth=2) # plot the loss
pt = compare_trained_and_untrained_flow(flow_trained_BN, flow_untrained_BN, p, 1000)
plot!(; xlims=(-50, 50), ylims=(-100, 20))

testmode!(flow_trained_BN)

xs = randn(Float32, 4, 10)
ys = flow_trained_BN.transform(xs)
xs0 = inverse(flow_trained_BN.transform)(ys)
xs0 .- xs

samples_trained = rand_batch(flow_trained_BN, n_samples)
samples_untrained = rand_batch(flow_untrained_BN, n_samples)
samples_true = rand(p, n_samples)

# evaluate sample error
testmode!(flow_trained_BN)
flow1 = deepcopy(flow_trained_BN)
flow2 = deepcopy(flow_trained_BN)

using Functors: fmap
using BigFloat
setprecision(BigFloat, 256)
ft = BigFloat

@functor MvNormal
flow_big = Flux._paramtype(BigFloat, flow2)

Xs = randn(Float32, 4, 1000)
Xs_big = ft.(Xs)
ts = flow1.transform
ts_big = flow_big.transform

ts(Xs) .- ts_big(Xs_big)

# density error
Ys = rand(p, 1000)
Ys_big = ft.(Ys)

Zs = ts(Xs)
Zs_big = ft.(Zs)

logpdf(flow1, Zs) .- logpdf(flow_big, Zs)

.-logpdf(flow_big, Ys_big)