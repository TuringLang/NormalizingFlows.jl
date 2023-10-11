using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
include("../../../common.jl")
include("../../hamiltonian_layer.jl")
include("../../../MLP/invertibleMLP.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../../../targets/banana.jl")

# create target p
p = Banana(2, 3.0f-1, 100.0f0)
# samples = rand(p, 1000)
# visualize(p, samples)

######################################
# setup flow
######################################
logp = Base.Fix1(logpdf, p)
∇S = Base.Fix1(Score, p)
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

dims = p.dim
L = 200
nlayers = 6
maps = [
    [
        LeapFrog(dims, log(1.0f-2), L, ∇S, ∇logm),
        InvertibleMLP(2 * dims),
        Flux._paramtype(Float32, PlanarLayer(2 * dims)),
    ] for i in 1:nlayers
]
Ls = reduce(vcat, maps)
ts = fchain(Ls)
q0 = MvNormal(zeros(Float32, 2dims), I)
flow = Bijectors.transformed(q0, ts)
flow_untrained = deepcopy(flow)

θ, re = Optimisers.destructure(flow_untrained)

######################################
# setup big flow
######################################
setprecision(BigFloat, 2048)
bf = BigFloat

# ts = flow_trained.transform # extract trained transformation
# its = inverse(ts) # extract trained inverse transformation

p_big = Banana(2, bf(3.0f-1), bf(100.0f0))
∇S_big = Base.Fix1(Score, p_big)
maps_big = [
    [
        LeapFrog(dims, log(bf(1.0f-2)), L, ∇S_big, ∇logm),
        InvertibleMLP(2 * dims),
        PlanarLayer(2 * dims),
    ] for i in 1:6
]
Ls_big = reduce(vcat, maps_big)
ts_untrained_big = Flux._paramtype(bf, fchain(Ls_big))

θ_big, re_big = Flux.destructure(ts_untrained_big)
# θ_trained, re_after = Flux.destructure(ts) # extract trained parameters 

# ts_big = re(bf.(θ_trained)) # construct big transformation
# its_big = inverse(ts_big) # construct big inverse transformation

@functor MvNormal
q0_big = Flux._paramtype(bf, q0)
# flow_big = Bijectors.transformed(q0_big, ts_big) # construct big flow
@functor MvNormal ()

##################################
# make dir
############################
if !isdir("figure")
    mkdir("figure")
end
if !isdir("result")
    mkdir("result")
end
