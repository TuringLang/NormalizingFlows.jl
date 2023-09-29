using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
using Flux: f32
using Plots
include("../common.jl")
include("hamiltonian_layer.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../targets/banana.jl")

# create target p
p = Banana(2, 5.0f-1, 100.0f0)
samples = rand(p, 1000)
visualize(p, samples)

logp = Base.Fix1(logpdf, p)
∇S = Base.Fix1(Score, p)
∇logm(x) = -x # gaussian momentum
function logp_joint(z)
    dim = div(length(z), 2)
    x, ρ = z[1:dim], z[(dim + 1):end]
    return logp(x) + logpdf(MvNormal(zeros(Float32, dims), I), ρ)
end

dims = p.dim
L = 20
Ls = [
    Bijectors.Scale(ones(Float32, 2dims)) ∘ Bijectors.Shift(zeros(Float32, 2dims)) ∘
    LeapFrog(dims, log(1.0f-2), 1.0f0, L, ∇S, ∇logm) for i in 1:3
]
q0 = MvNormal(zeros(Float32, 2dims), I)
flow = Bijectors.transformed(q0, ∘(Ls...))
# flow = Bijectors.transformed(q0, trans)
flow_untrained = deepcopy(flow)

sample_per_iter = 5
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp_joint,
    sample_per_iter;
    max_iters=200_00,
    optimiser=Optimisers.Adam(1e-3),
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses = map(x -> x.loss, stats)

plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, p, 1000)

#####################333
# test stability
#####################333
using Functors: fmap
setprecision(BigFloat, 256)
ft = BigFloat

@functor MvNormal ()
flow_big = Flux._paramtype(BigFloat, flow_trained)

Xs = randn(Float32, 2dims, 1000)
Xs_big = ft.(Xs)
ts = flow_trained.transform
ts_big = flow_big.transform

Xs_ts = hcat(map(ts, eachcol(Xs))...)
Xs_ts_big = hcat(map(ts_big, eachcol(Xs_big))...)
Xs_ts .- Xs_ts_big

# density error
Ys = rand(p, 1000)
Ys_big = ft.(Ys)

i = rand(1:1000)
logpdf(flow_trained, Xs_ts[:, i]) .- logpdf(flow_big, Xs_ts_big[:, i])