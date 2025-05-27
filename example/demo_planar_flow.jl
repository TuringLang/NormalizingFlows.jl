using Random, Distributions, LinearAlgebra, Bijectors
using Functors
using Optimisers, ADTypes, Mooncake
using NormalizingFlows

include("SyntheticTargets.jl")
include("utils.jl")

Random.seed!(123)
rng = Random.default_rng()
T = Float64

######################################
# 2d Banana as the target distribution
######################################
target = Banana(2, 1.0, 10.0)
logp = Base.Fix1(logpdf, target)


######################################
# setup planar flow
######################################
function create_planar_flow(n_layers::Int, q₀)
    d = length(q₀)
    Ls = [PlanarLayer(d) for _ in 1:n_layers]
    ts = reduce(∘, Ls)
    return transformed(q₀, ts)
end

@leaf MvNormal
q0 = MvNormal(zeros(T, 2), ones(T, 2))
flow = create_planar_flow(10, q0)
flow_untrained = deepcopy(flow)

######################################
# start training
######################################
sample_per_iter = 32

# callback function to log training progress
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
adtype = ADTypes.AutoMooncake(; config = Mooncake.Config())
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
flow_trained, stats, _ = train_flow(
    elbo_batch,
    flow,
    logp,
    sample_per_iter;
    max_iters=10_000,
    optimiser=Optimisers.Adam(one(T)/100),
    ADbackend=adtype,
    show_progress=true,
    callback=cb,
    hasconverged=checkconv,
)
θ, re = Optimisers.destructure(flow_trained)
losses = map(x -> x.loss, stats)

######################################
# evaluate trained flow
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, target, 1000)
