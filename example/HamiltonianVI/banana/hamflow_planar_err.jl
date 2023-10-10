using Random, Distributions, LinearAlgebra, Bijectors
using Optimisers
using FunctionChains
using JLD2
using Functors
include("../../common.jl")
include("../hamiltonian_layer.jl")
include("../../MLP/invertibleMLP.jl")

Random.seed!(123)
rng = Random.default_rng()

######################################
# 2d Banana as the target distribution
######################################
include("../../targets/banana.jl")

# read  target p and trained flow
res = JLD2.load("res/ham_flow_planar.jld2")
p = res["target"]
flow_trained = res["model"]
L = 200
# samples = rand(p, 1000)
# visualize(p, samples)

# compare_trained_and_untrained_flow_BN(flow_trained, flow_trained, p, 1000)
#####################333
# construct bigfloat transformations
#####################333
setprecision(BigFloat, 2048)
ft = BigFloat

ts = flow_trained.transform # extract trained transformation
its = inverse(ts) # extract trained inverse transformation

dims = p_big.dim
maps = [
    [
        LeapFrog(dims, log(ft(1.0f-2)), L, ∇S_big, ∇logm),
        InvertibleMLP(2 * dims),
        PlanarLayer(2 * dims),
    ] for i in 1:5
]
Ls_big = reduce(vcat, maps)
ts_untrained_big = Flux._paramtype(ft, fchain(Ls_big))

θ, re = Flux.destructure(ts_untrained_big)
θ_trained, re_after = Flux.destructure(ts) # extract trained parameters 

ts_big = re(ft.(θ_trained)) # construct big transformation
its_big = inverse(ts_big) # construct big inverse transformation

q0 = flow_trained.dist # extract initial dist
@functor MvNormal
flow_big = Bijectors.transformed(Flux._paramtype(ft, q0), ts_big) # construct big flow

#####################
# test stability
######################
# this is how to extract functions 
function get_functions(ts)
    fs = FunctionChains._flatten_composed(ts)[1]
    return fs.fs
end

# forward sample stability
Xs = randn(Float32, 2dims, 1000)
Xs_big = ft.(Xs)

# # check stability of big flow
# Xs_big .- its_big(ts_big(Xs_big))
diff = ts(Xs) .- ts_big(Xs_big)
dd = Float32.(map(norm, eachcol(diff)))

fwd_sample = with_intermediate_results(ts, Xs)
fwd_sample_big = map(x -> Float32.(x), with_intermediate_results(ts_big, Xs_big))
fwd_diff_layer = fwd_sample .- fwd_sample_big
fwd_err_layer = map(x -> map(norm, eachcol(x)), fwd_diff_layer)

# sample error scaling
f1(x) = abs.(x)
f2(x) = sin.(x) .+ 1
f3(x) = 1 ./ (1 .+ exp.(-x))
stat_layer = map(x -> map(f1, x), fwd_err_layer)

# bwd stability
# density error
Ys = ts(Xs)
# Ys = vcat(rand(p, 1000), randn(Float32, 2, 1000))
Ys_big = ft.(Ys)
diff_inv = its(Ys) .- its_big(Ys_big)
dd_inv = Float32.(map(norm, eachcol(diff_inv)))

err_lpdf = Float32.(abs.(logpdf(flow_trained, Ys) .- logpdf(flow_big, Ys_big)))
err_lpdf_rel =
    Float32.(
        abs.(logpdf(flow_trained, Ys) .- logpdf(flow_big, Ys_big)) ./
        abs.(logpdf(flow_big, Ys_big))
    )

# no density err is too big for the Ham flow stability ---
# show the shadoing window is huge

#####################
# elbo err 
#####################
