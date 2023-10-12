using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows
using Zygote
using Flux
using JLD2
include("../../common.jl")
include("../invertibleMLP.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../../targets/banana.jl")

# create target p
p = Banana(2, 3.0f-1, 100.0f0)
logp = Base.Fix1(logpdf, p)

Data = rand(p, 60000)
data_load = Flux.DataLoader(Data; batchsize=200, shuffle=true)

######################################
# construct the flow
######################################
d = p.dim
μ = zeros(Float32, d)
Σ = I

nlayers = 15
maps = [
    [
        InvertibleMLP(d),
        Flux._paramtype(Float32, PlanarLayer(2)),
        Flux._paramtype(Float32, RadialLayer(2)),
    ] for i in 1:nlayers
]
Ls = reduce(vcat, maps)
ts = fchain(Ls)
q0 = MvNormal(μ, Σ)
flow = transformed(q0, ts)

flow_untrained = deepcopy(flow)
θ, re = Optimisers.destructure(flow_untrained)
#####################################3
# change precision
#####################################
function set_precision_flow(ft::DataType, θ_trained, q0)
    maps_new = [
        [
            InvertibleMLP(d),
            Flux._paramtype(ft, PlanarLayer(2)),
            Flux._paramtype(ft, RadialLayer(2)),
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
    return flow_new, ts_new, its_new, q0_new, re_new
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
