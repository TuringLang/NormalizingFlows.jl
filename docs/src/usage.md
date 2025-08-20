## General usage

`train_flow` is the main function to train a normalizing flow. 
The users mostly need to specify a normalizing flow `flow`, 
the variational objective `vo` and its corresponding arguments `args...`.

```@docs
NormalizingFlows.train_flow
```

The flow object can be constructed by `transformed` function in `Bijectors.jl`.
For example, for mean-field Gaussian VI, we can construct the flow family as follows:

```julia
using Distributions, Bijectors
T = Float32
@leaf MvNormal # to prevent params in q₀ from being optimized
q₀ = MvNormal(zeros(T, 2), ones(T, 2))
# the flow family is defined by a shift and a scale 
flow = Bijectors.transformed(q₀, Bijectors.Shift(zeros(T,2)) ∘ Bijectors.Scale(ones(T, 2)))
```

To train the Gaussian VI targeting distribution `p` via ELBO maximization, run:

```julia
using NormalizingFlows, Optimisers
using ADTypes, Mooncake

sample_per_iter = 10
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters=5_000,
    optimiser=Optimisers.Adam(one(T)/100),
    ADbackend=ADTypes.AutoMooncake(; config=Mooncake.Config()),
    show_progress=true,
)
```