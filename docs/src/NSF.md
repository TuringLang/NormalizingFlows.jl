# Demo of NSF on 2D Banana Distribution

```julia
using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Zygote
using NormalizingFlows


target = Banana(2, one(T), 100one(T))
logp = Base.Fix1(logpdf, target)

######################################
# learn the target using Neural Spline Flow
######################################
@leaf MvNormal
q0 = MvNormal(zeros(T, 2), I)


flow = nsf(q0; paramtype=T)
flow_untrained = deepcopy(flow)
######################################
# start training
######################################
sample_per_iter = 64

# callback function to log training progress
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
# nsf only supports AutoZygote
adtype = ADTypes.AutoZygote()
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
flow_trained, stats, _ = train_flow(
    elbo_batch,
    flow,
    logp,
    sample_per_iter;
    max_iters=10,   # change to larger number of iterations (e.g., 50_000) for better results
    optimiser=Optimisers.Adam(1e-4),
    ADbackend=adtype,
    show_progress=true,
    callback=cb,
    hasconverged=checkconv,
)
θ, re = Optimisers.destructure(flow_trained)
losses = map(x -> x.loss, stats)
```