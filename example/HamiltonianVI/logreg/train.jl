using JLD2
include("setup.jl")

# just need to pass a large batch of  samples through the flow
Ys = rand_batch(flow, 10_000)

param_trained, re = Optimisers.destructure(flow)

JLD2.save("result/hamflow.jld2", "param", param_trained, "L", L, "nlayers", nlayers)