using JLD2
include("setup.jl")

# just need to pass a large batch of  samples through the flow
Ys = rand_batch(flow, 30_000)

# check_trained_flow(flow, p, 1000)
param_trained, re = Optimisers.destructure(flow)

JLD2.save(
    "result/hamflow.jld2",
    "flow",
    flow,
    "param",
    param_trained,
    "L",
    L,
    "nlayers",
    nlayers,
    "target",
    p,
)