using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
using Flux: f32
using Plots
include("../common.jl")

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
# learn the target using radial flow 
######################################
function create_radial_flow(n_layers::Int, q₀)
    d = length(q₀)
    Ls = [f32(RadialLayer(d)) for _ in 1:n_layers]
    ts = fchain(Ls)
    return transformed(q₀, ts)
end

# create a 20-layer radial flow
flow = create_radial_flow(10, MvNormal(zeros(Float32, 2), I))
flow_untrained = deepcopy(flow)

# train the flow
sample_per_iter = 10
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters=200_00,
    optimiser=Optimisers.ADAM(),
    callback=cb,
)
losses = map(x -> x.loss, stats)

######################################
# evaluate trained flow
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, p, 1000; legend=:bottom)
