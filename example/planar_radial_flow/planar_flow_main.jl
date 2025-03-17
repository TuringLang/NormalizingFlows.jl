using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using NormalizingFlows
using Mooncake
using CUDA
using Flux: f32
using Flux
using Plots
include("common.jl")

Random.seed!(123)
rng = Random.default_rng()
T = Float32

######################################
# 2d Banana as the target distribution
######################################
include("targets/banana.jl")

# create target p
p = Banana(2, 1.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

######################################
# learn the target using planar flow 
######################################
function create_planar_flow(n_layers::Int, q₀)
    d = length(q₀)
    Ls = [f32(PlanarLayer(d)) for _ in 1:n_layers]
    ts = reduce(∘, Ls)
    return transformed(q₀, ts)
end

# create a 10-layer planar flow
q0 = MvNormal(zeros(T, 2), ones(T, 2))
flow = create_planar_flow(10, q0)



# train the flow
sample_per_iter = 10
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
flow_trained, stats, _ = train_flow(
    elbo,
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
# evaluate trained flow
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, p, 1000)
