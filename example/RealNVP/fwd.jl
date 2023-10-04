using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
using Flux
using Plots
include("../common.jl")
include("AffineCoupling.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../targets/banana.jl")

# create target p
p = Banana(4, 1.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

Data = rand(p, 10000)
data_load = Flux.DataLoader(Data; batchsize=50, shuffle=true)

######################################
# learn the target using Affine coupling flow
######################################
# μ = vec(mean(Data; dims=2))
# Σ = vec(var(Data; dims=2))
d = p.dim
μ = zeros(Float32, d)
Σ = I

hdims = 4
Ls = [AffineCoupling(d, hdims, 1:2) ∘ AffineCoupling(d, hdims, 3:4) for i in 1:3]
ts = ∘(Ls...)
q0 = MvNormal(μ, Σ)
flow = transformed(q0, ts)
flow_untrained = deepcopy(flow)

Ls_res = [AffineCouplingRes(d, hdims, 1:2) ∘ AffineCouplingRes(d, hdims, 3:4) for i in 1:3]
ts_res = ∘(Ls_res...)
flow_res = transformed(q0, ts_res)
flow_res_untrained = deepcopy(flow_res)

#########################################
# samples from the target 
##########################################

# train the flow
function train(flow, data_load, opt, n_epoch=1000)
    # destruct flow for explicit access to the parameters
    # use FunctionChains instead of simple compositions to construct the flow when many flow layers are involved
    # otherwise the compilation time for destructure will be too long
    θ_flat, re = Optimisers.destructure(flow)

    # Normalizing flow training loop 
    θ_flat_trained, opt_stats, st = NormalizingFlows.optimize(
        data_load, AutoZygote(), llh_batch, θ_flat, re; n_epoch=n_epoch, optimiser=opt
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, opt_stats, st
end

# flow_trained, stats, _ = train(flow, data_load, Optimisers.ADAM(1e-3), 1000)
flow_res_trained, stats_res, _ = train(flow_res, data_load, Optimisers.ADAM(1e-3), 1000)
# losses = map(x -> x.loss, stats_res)

θ, re = Optimisers.destructure(flow)
loss(θ_) = -llh_batch(re(θ_), rand(p, 20))

pt = compare_trained_and_untrained_flow(flow_res_trained, flow_res_untrained, p, 1000)
plot!(; xlims=(-50, 50), ylims=(-100, 20))