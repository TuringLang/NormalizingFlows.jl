using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
include("../../common.jl")
include("../hamiltonian_layer.jl")
include("../momentum.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../../targets/banana.jl")

include("model.jl")

######################################
# setup flow
######################################
∇logm(x) = -x # gaussian momentum
function logp_joint(z::AbstractVector{T}) where {T}
    dim = div(length(z), 2)
    x, ρ = z[1:dim], z[(dim + 1):end]
    return logp(x) + logpdf(MvNormal(zeros(eltype(z), dim), I), ρ)
end
function logp_joint(zs::AbstractMatrix{T}) where {T}
    dim = div(size(zs, 1), 2)
    xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]
    return logp(xs) + logpdf(MvNormal(zeros(eltype(zs), dim), I), ρs)
end

dims = d # defined in model
L = 60
nlayers = 150
stepsize = 2e-2
maps = [
    [LeapFrog(dims, log(stepsize), L, ∇S, ∇logm), MomentumNorm(dims)] for i in 1:nlayers
]
Ls = reduce(vcat, maps)
ts = fchain(Ls)
q0 = MvNormal(zeros(2dims), I)
flow = Bijectors.transformed(q0, ts)
flow_untrained = deepcopy(flow)

θ, re = Optimisers.destructure(flow_untrained)

######################################
# setup big flow
######################################
function set_precision_flow(ft::DataType, θ_trained, q0)
    logp_new(x) = logpost(x, ft.(X), ft.(Y), ft(aa), ft(bb), p)

    ∇S_new(x) = ∇logp(x, ft.(X), ft.(Y), ft(aa), ft(bb), p)
    maps_new = [
        [
            LeapFrog(dims, log(ft(stepsize)), L, ∇S_new, ∇logm),
            MomentumNorm(
                dims, zeros(ft, dims), randn(ft, dims, dims), randn(ft, dims, dims)
            ),
        ] for i in 1:nlayers
    ]
    Ls_new = reduce(vcat, maps_new)
    ts_untrained_new = Flux._paramtype(ft, fchain(Ls_new))

    θ_, re_new = Optimisers.destructure(ts_untrained_new)
    @functor MvNormal
    q0_new = Flux._paramtype(ft, q0)
    @functor MvNormal ()
    # construct new ts, itsm and flow
    ts_new = re_new(ft.(θ_trained))
    its_new = inverse(ts_new)
    flow_new = Bijectors.transformed(q0_new, ts_new)
    return flow_new, ts_new, its_new, q0_new, logp_new, re_new
end
##################################
# make dir
############################
if !isdir("figure")
    mkdir("figure")
end
if !isdir("result")
    mkdir("result")
end
