using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using InvertibleNetworks
using Zygote
using Flux
using Plots
using ProgressMeter, IterTools
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

Data = rand(p, 100000)
Data = reshape(Data, (1, 1, size(Data)...))
data_load = Flux.DataLoader(Data; batchsize=150, shuffle=true)

#copy directly from tutorial: https://transform.softwareunderground.org/2022-julia-for-geoscience/normalizing-flow-training
function loss_inveritble_networks(G, X)
    batch_size = size(X)[end]

    Z, lgdet = G.forward(X)

    l2_loss = 0.5 * norm(Z)^2 / batch_size  #likelihood under Normal Gaussian training 
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
flowsteps_K = 20

G = NetworkGlow(n_in, n_hidden, levels_L, flowsteps_K;)

lr = 9.0f-4
opt = Flux.ADAM(lr)

nnls, _, _ = train_invertible_networks!(G, loss, data_load, 50, opt)

num_test_samples = 500;
Z_test = randn(Float32, nx, ny, n_in, num_test_samples);
Zns = G.inverse(Z_test)

Ys = reshape(Zns, size(Zns)[3:4])

p = scatter(
    Ys[1, :], Ys[2, :]; label="True Distribution", color=:blue, markersize=2, alpha=0.5
)

xlabel!(p, "X")
ylabel!(p, "Y")