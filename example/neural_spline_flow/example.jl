using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
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

#  work in progress, not for review