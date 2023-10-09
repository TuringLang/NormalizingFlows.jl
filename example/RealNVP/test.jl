using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using InvertibleNetworks
using Zygote
using Flux
using Plots
using ProgressMeter, IterTools
include("../common.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../targets/banana.jl")

# create target p
p = Banana(2, 5.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

Data = rand(p, 120000)
Data = reshape(Data, (1, 1, size(Data)...))
data_load = Flux.DataLoader(Data; batchsize=150, shuffle=true)

# make data matrix into 4way tensor
adddim(M) = reshape(M, (1, 1, size(M)...))
# bring empty 4way tensor back to mat
rmdim(M) = reshape(M, size(M)[3:4])

#copy directly from tutorial: https://transform.softwareunderground.org/2022-julia-for-geoscience/normalizing-flow-training
function loss(G, X)
    batchsize = size(X)[end]

    Y, logdet = G.forward(X)
    f = 0.5f0 / batchsize * norm(Y)^2 - logdet
    ΔX, X_ = G.backward(1.0f0 ./ batchsize * Y, Y)

    return f
end

#network architecture
nx = 1
ny = 1
n_in = 2 #put 2d variables into 2 channels
n_hidden = 16
levels_L = 1
flowsteps_K = 5

G = NetworkGlow(n_in, n_hidden, levels_L, flowsteps_K)

lr = 5.0f-4
opt = Flux.ADAM(lr)

nnls = train_invertible_networks!(G, loss, data_load, 10, opt)

# show learned flow
num_test_samples = 1000;
Z_test = randn(Float32, nx, ny, n_in, num_test_samples);
Zns = G.inverse(Z_test)
Ys = reshape(Zns, size(Zns)[3:4])

p = scatter(Ys[1, :], Ys[2, :]; label="trained flow", color=:blue, markersize=2, alpha=0.5)

xlabel!(p, "X")
ylabel!(p, "Y")

using JLD2
JLD2.save(
    "res/banana_50.jld2",
    "G",
    G,
    "nx",
    nx,
    "ny",
    ny,
    "n_in",
    n_in,
    "n_hidden",
    n_hidden,
    "levels_L",
    levels_L,
    "flowsteps_K",
    flowsteps_K,
    "lr",
    lr,
    "nnls",
    nnls,
)

####################333
# examine stability
####################

setprecision(BigFloat, 256)
ft = BigFloat

θ, re = Optimisers.destructure(G)
G1 = Flux._paramtype(BigFloat, G)
θ1, re1 = Optimisers.destructure(G1)

ts(xs) = rmdim(G.inverse(adddim(xs)))
ts1(xs) = rmdim(G1.inverse(adddim(xs)))

its(xs) = rmdim(G.forward(adddim(xs))[1])
its1(xs) = rmdim(G1.forward(adddim(xs))[1])

num_test_samples = 1000;
xs = randn(Float32, n_in, num_test_samples)
xs_big = ft.(xs)

diffs = xs .- its(ts(xs))
dd = map(norm, eachcol(diffs))

using JLD2
JLD2.save("res/invertiblenet.jld2", "G", G, "param", θ, "dd", dd)

# load the saved model
res = JLD2.load("res/invertiblenet.jld2")