using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
include("../../common.jl")
include("../hamiltonian_layer.jl")
include("../../MLP/invertibleMLP.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../../targets/banana.jl")

# create target p
p = Banana(2, 3.0f-1, 100.0f0)
# samples = rand(p, 1000)
# visualize(p, samples)

logp = Base.Fix1(logpdf, p)
∇S = Base.Fix1(Score, p)
∇logm(x) = -x # gaussian momentum
function logp_joint(z::AbstractVector{T}) where {T}
    dim = div(length(z), 2)
    x, ρ = z[1:dim], z[(dim + 1):end]
    return logp(x) + logpdf(MvNormal(zeros(eltype(z), dim), I), ρ)
end
function logp_joint(zs::AbstractMatrix{T}) where {T}
    dim = div(size(zs, 1), 2)
    xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]
    return logp(xs) + logpdf(MvNormal(zeros(eltype(zs), dim), I), ρs)
end

dims = p.dim
L = 200
nlayers = 10
maps = [
    [LeapFrog(dims, log(1.0f-2), L, ∇S, ∇logm), InvertibleMLP(2 * dims)] for i in 1:nlayers
]
Ls = reduce(vcat, maps)
ts = fchain(Ls)
q0 = MvNormal(zeros(Float32, 2dims), I)
flow = Bijectors.transformed(q0, ts)
flow_untrained = deepcopy(flow)

sample_per_iter = 30
max_iters = 200_000
opt = Optimisers.Adam(1.0f-4)
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1.0f-3
flow_trained, stats, _ = train_flow(
    elbo_batch,
    flow,
    logp_joint,
    sample_per_iter;
    max_iters=max_iters,
    optimiser=opt,
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses = map(x -> x.loss, stats)

param_trained, re = Optimisers.destructure(flow_trained)

using JLD2
JLD2.save(
    "res/ham_flow_big.jld2",
    "flow",
    flow_trained,
    "param",
    param_trained,
    "L",
    L,
    "elbo",
    losses,
    "stat",
    stats,
    "target",
    p,
    "max_iters",
    max_iters,
    "opt",
    opt,
    "batch_size",
    sample_per_iter,
)
# res = JLD2.load("res/big_ham.jld2")
# flow_trained = res["model"]

# plot(losses; label="Loss", linewidth=2) # plot the loss
# compare_trained_and_untrained_flow_BN(flow_trained, flow_untrained, p, 1000)
