using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Mooncake
using NormalizingFlows

include("SyntheticTargets.jl")
include("utils.jl")

##################################
# start demo
#################################
Random.seed!(123)
rng = Random.default_rng()
T = Float32

######################################
# a difficult banana target
######################################
target = Banana(2, one(T), 100one(T))
logp = Base.Fix1(logpdf, target)


######################################
# learn the target using Affine coupling flow
######################################
@leaf MvNormal
q0 = MvNormal(zeros(T, 2), I)

d = 2
hdims = [16, 16]
nlayers = 3

# use NormalizingFlows.realnvp to create a RealNVP flow
flow = realnvp(q0, hdims, nlayers; paramtype=T)
flow_untrained = deepcopy(flow)


######################################
# start training
######################################
sample_per_iter = 16

# callback function to log training progress
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
adtype = ADTypes.AutoMooncake(; config = Mooncake.Config())

checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
flow_trained, stats, _ = train_flow(
    rng, 
    elbo_batch,        # using elbo_batch instead of elbo achieves 4-5 times speedup 
    flow,
    logp,
    sample_per_iter;
    max_iters=10,   # change to larger number of iterations (e.g., 50_000) for better results
    optimiser=Optimisers.Adam(5e-4),
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
