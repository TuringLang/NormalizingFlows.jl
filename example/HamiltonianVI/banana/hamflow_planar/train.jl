using JLD2
include("setup.jl")

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

JLD2.save(
    "result/hamflow_planar.jld2",
    "flow",
    flow_trained,
    "param",
    param_trained,
    "L",
    L,
    "nlayers",
    nlayers,
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