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

Data = rand(p, 100000)
Data = reshape(Data, (1, 1, size(Data)...))
data_load = Flux.DataLoader(Data; batchsize=200, shuffle=true)

#copy directly from tutorial: https://transform.softwareunderground.org/2022-julia-for-geoscience/normalizing-flow-training
function loss_invertible_networks(G, X)
    batch_size = size(X)[end]

    Z, lgdet = G.forward(X)

    l2_loss = 0.5 * norm(Z)^2 / batch_size #likelihood under Normal Gaussian training 
    dZ = Z / batch_size                   #gradient under Normal Gaussian training

    G.backward(dZ, Z)  #sets gradients of G wrt output and also logdet terms

    return (l2_loss, lgdet)
end

#network architecture
nx = 1
ny = 1
n_in = 2 #put 2d variables into 2 channels
n_hidden = 16
levels_L = 1
flowsteps_K = 40

G = NetworkGlow(n_in, n_hidden, levels_L, flowsteps_K;)

lr = 9.0f-4
opt = Flux.ADAM(lr)

nnls, _, _ = train_invertible_networks!(G, loss_invertible_networks, data_load, 100, opt)

####################333
# examine stability
####################
num_test_samples = 500;
Z_test = randn(Float32, nx, ny, n_in, num_test_samples);
Zns = G.inverse(Z_test)
Ys = reshape(Zns, size(Zns)[3:4])

# make data matrix into 4way tensor
adddim(M) = reshape(M, (1, 1, size(M)...))
# bring empty 4way tensor back to mat
rmdim(M) = reshape(M, size(M)[3:4])

p = scatter(
    Ys[1, :], Ys[2, :]; label="True Distribution", color=:blue, markersize=2, alpha=0.5
)

xlabel!(p, "X")
ylabel!(p, "Y")

setprecision(BigFloat, 256)
ft = BigFloat

G1 = Flux._paramtype(BigFloat, G)
θ1, re1 = Optimisers.destructure(G1)

ts(xs) = rmdim(G.inverse(adddim(xs)))
ts1(xs) = rmdim(G1.inverse(adddim(xs)))

its(xs) = rmdim(G.forward(adddim(xs))[1])
its1(xs) = rmdim(G1.forward(adddim(xs))[1])

num_test_samples = 500;
xs = randn(Float32, n_in, num_test_samples)
xs_big = ft.(xs)

norm(xs .- its(ts(xs)))
