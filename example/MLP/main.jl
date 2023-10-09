using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
using Flux
using Plots
include("../common.jl")
include("invertibleMLP.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../targets/banana.jl")

# create target p
p = Banana(2, 5.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

Data = rand(p, 20000)
data_load = Flux.DataLoader(Data; batchsize=200, shuffle=true)

######################################
# construct the flow
######################################
d = p.dim
μ = zeros(Float32, d)
Σ = I

Ls = [InvertibleMLP(d) ∘ Flux._paramtype(Float32, PlanarLayer(2)) for i in 1:20]
ts = fchain(Ls)
q0 = MvNormal(μ, Σ)
flow = transformed(q0, ts)
flow_untrained = deepcopy(flow)
######################
# train the flow
#####################
function train(flow, data_load, opt, n_epoch=100)
    # destruct flow for explicit access to the parameters
    # use FunctionChains instead of simple compositions to construct the flow when many flow layers are involved
    # otherwise the compilation time for destructure will be too long
    θ_flat, re = Optimisers.destructure(flow)

    # Normalizing flow training loop 
    θ_flat_trained, opt_stats, st = NormalizingFlows.optimize(
        data_load,
        AutoZygote(),
        NormalizingFlows.llh_batch,
        θ_flat,
        re;
        n_epoch=n_epoch,
        optimiser=opt,
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, opt_stats, st
end

flow_trained, stats, _ = train(flow, data_load, Optimisers.ADAM(1e-3), 1000)
pt = compare_trained_and_untrained_flow_BN(flow_trained, flow_untrained, p, 1000)
plot!(; xlims=(-50, 50), ylims=(-100, 20))

# stability
setprecision(BigFloat, 256)
ft = BigFloat

@functor MvNormal
flow_big = Flux._paramtype(BigFloat, flow_trained)

Xs = randn(Float32, 2, 1000)
Xs_big = ft.(Xs)
ts = flow_trained.transform
ts_big = flow_big.transform
its = inverse(flow_trained.transform)
its_big = inverse(flow_big.transform)

diff = ts(Xs) .- ts_big(Xs_big)
dd = map(norm, eachcol(diff))

# density error

Ys = rand(p, 1000)
Ys_big = ft.(Ys)

diff_inv = its(Ys) .- its_big(Ys_big)

logpdf(flow_trained, Ys) .- logpdf(flow_big, Ys_big)
