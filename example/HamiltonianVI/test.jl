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
include("../MLP/invertibleMLP.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../targets/banana.jl")

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
L = 250
Ls = [LeapFrog(dims, log(1.0f-2), L, ∇S, ∇logm) ∘ InvertibleMLP(2 * dims) for i in 1:5]
ts = ∘(Ls...)
q0 = MvNormal(zeros(Float32, 2dims), I)
flow = Bijectors.transformed(q0, ∘(Ls...))
# flow = Bijectors.transformed(q0, trans)
flow_untrained = deepcopy(flow)

sample_per_iter = 30
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1.0f-3
flow_trained, stats, _ = train_flow(
    elbo_batch,
    flow,
    logp_joint,
    sample_per_iter;
    max_iters=1000_00,
    optimiser=Optimisers.Adam(1.0f-4),
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses = map(x -> x.loss, stats)

using JLD2
JLD2.save("res/big_ham.jld2", "model", flow_trained, "elbo", losses)

res = JLD2.load("res/big_ham.jld2")

# plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow_BN(flow_trained, flow_untrained, p, 1000)

# #####################333
# # test stability
# #####################333
using Functors: fmap
setprecision(BigFloat, 256)
ft = BigFloat

@functor MvNormal
flow_big = Flux._paramtype(BigFloat, flow_trained)

Xs = randn(Float32, 2dims, 1000)
Xs_big = ft.(Xs)
ts = flow_trained.transform
ts_big = flow_big.transform
its, its_big = inverse(ts), inverse(ts_big)

diff = ts(Xs) .- ts_big(Xs_big)
dd = map(norm, eachcol(diff))

# # density error
Ys = ts(Xs)
Ys_big = ft.(Ys)
diff_inv = its(Ys) .- its_big(Ys_big)
dd_inv = map(norm, eachcol(diff_inv))

logpdf(flow_trained, Ys) .- logpdf(flow_big, Ys_big)
# i = rand(1:1000)