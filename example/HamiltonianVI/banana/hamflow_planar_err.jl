using Random, Distributions, LinearAlgebra, Bijectors
using Optimisers
using FunctionChains
using JLD2
using Functors
using NormalizingFlows
using Base.Threads
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

compare_trained_and_untrained_flow_BN(flow_trained, flow_trained, p, 1000)
#####################333
# construct bigfloat transformations
#####################333
setprecision(BigFloat, 2048)
ft = BigFloat

ts = flow_trained.transform # extract trained transformation
its = inverse(ts) # extract trained inverse transformation

p_big = Banana(2, ft(3.0f-1), ft(100.0f0))
∇S_big = Base.Fix1(Score, p_big)
dims = p_big.dim
maps = [
    [
        LeapFrog(dims, log(ft(1.0f-2)), L, ∇S_big, ∇logm),
        InvertibleMLP(2 * dims),
        PlanarLayer(2 * dims),
    ] for i in 1:6
]
Ls_big = reduce(vcat, maps)
ts_untrained_big = Flux._paramtype(ft, fchain(Ls_big))

θ, re = Flux.destructure(ts_untrained_big)
θ_trained, re_after = Flux.destructure(ts) # extract trained parameters 

ts_big = re(ft.(θ_trained)) # construct big transformation
its_big = inverse(ts_big) # construct big inverse transformation

q0 = flow_trained.dist # extract initial dist
@functor MvNormal
q0_big = Flux._paramtype(ft, q0)
flow_big = Bijectors.transformed(q0_big, ts_big) # construct big flow

#####################
# test stability
######################
# this is how to extract functions 
function get_functions(ts)
    fs = FunctionChains._flatten_composed(ts)[1]
    return fs.fs
end

function intermediate_flows(ts, q0)
    flows = []
    fs = get_functions(ts)
    for i in 1:length(fs)
        push!(flows, Bijectors.transformed(q0, fchain(fs[1:i])))
    end
    return flows
end
function intermediate_lpdfs(ts, q0, fwd_samples)
    flows = intermediate_flows(ts, q0)
    @assert length(flows) == length(fwd_samples) "numder of layers and numbers of sample batches mismatch"
    lpdfs = Vector{Vector{eltype(fwd_samples[1])}}(undef, length(flows))
    @threads for i in 1:length(flows)
        flow = flows[i]
        ys = fwd_samples[i]
        lpdf = logpdf(flow, ys)
        lpdfs[i] = lpdf
    end
    return reduce(hcat, lpdfs)
end
function inverse_from_intermediate_layers(ts, fwd_samples)
    inv_ts = []
    fs = get_functions(ts)
    for i in 1:length(fs)
        it = inverse(fchain(fs[1:i]))
        push!(inv_ts, it)
    end

    @assert length(inv_ts) == length(fwd_samples) "numder of layers and numbers of sample batches mismatch"
    X0 = Vector{Matrix{eltype(fwd_samples[1])}}(undef, length(inv_ts))
    @threads for i in 1:length(inv_ts)
        f = inv_ts[i]
        ys = fwd_samples[i]
        x0 = f(ys)
        X0[i] = x0
    end
    return X0
end
function elbo_intermediate(ts, q0, logp, Xs)
    flows = intermediate_flows(ts, q0)
    Els = Vector{eltype(Xs)}(undef, length(flows))
    @threads for i in 1:length(flows)
        flow = flows[i]
        el = elbo_batch(flow, logp, Xs)
        Els[i] = el
    end
    return Els
end

# forward sample stability
Xs = randn(Float32, 2dims, 1000)
Xs_big = ft.(Xs)

# # check stability of big flow
# Xs_big .- its_big(ts_big(Xs_big))
diff = ts(Xs) .- ts_big(Xs_big)
dd = Float32.(map(norm, eachcol(diff)))

fwd_sample = with_intermediate_results(ts, Xs)
# fwd_sample_big = map(x -> Float32.(x), with_intermediate_results(ts_big, Xs_big))
fwd_sample_big = with_intermediate_results(ts_big, Xs_big)
fwd_diff_layer = fwd_sample .- fwd_sample_big
fwd_err_layer = map(x -> map(norm, eachcol(x)), fwd_diff_layer)

#####################
# fwd sample error scaling
#####################
f1(x) = abs.(x)
f2(x) = sin.(x) .+ 1
f3(x) = 1 ./ (1 .+ exp.(-x))
s1(x) = sum(f1, x)
s2(x) = sum(f2, x)
s3(x) = sum(f3, x)
s1_layer = map(x -> mean(map(s1, eachcol(x))), fwd_sample)
s2_layer = map(x -> mean(map(s2, eachcol(x))), fwd_sample)
s3_layer = map(x -> mean(map(s3, eachcol(x))), fwd_sample)

s1_layer_big = map(x -> mean(map(s1, eachcol(x))), fwd_sample_big)
s2_layer_big = map(x -> mean(map(s2, eachcol(x))), fwd_sample_big)
s3_layer_big = map(x -> mean(map(s3, eachcol(x))), fwd_sample_big)

s1_layer_diff = abs.(s1_layer .- s1_layer_big)
s2_layer_diff = abs.(s2_layer .- s2_layer_big)
s3_layer_diff = abs.(s3_layer .- s3_layer_big)
#= small fwd err in general: should see small window size =#

#####################
# density error
#####################
Ys = ts(Xs)
# Ys = vcat(rand(p, 100), randn(Float32, 2, 100))
Ys_big = ft.(Ys)
diff_inv = its(Ys) .- its_big(Ys_big)
dd_inv = Float32.(map(norm, eachcol(diff_inv)))

bwd_sample = with_intermediate_results(its, Ys)
# bwd_sample_big = map(x -> Float32.(x), with_intermediate_results(its_big, Ys_big))
bwd_sample_big = with_intermediate_results(its_big, Ys_big)
bwd_diff_layer = bwd_sample .- bwd_sample_big
bwd_err_layer = map(x -> map(norm, eachcol(x)), bwd_diff_layer)

# err_lpdf = Float32.(abs.(logpdf(flow_trained, Ys) .- logpdf(flow_big, Ys_big)))

# err_lpdf_rel =
#     Float32.(
#         abs.(logpdf(flow_trained, Ys) .- logpdf(flow_big, Ys_big)) ./
#         abs.(logpdf(flow_big, Ys_big))
#     )

x0_layers = inverse_from_intermediate_layers(ts, map(x -> Float32.(x), fwd_sample_big))
x0_layers_big = map(
    x -> Float32.(x), inverse_from_intermediate_layers(ts_big, fwd_sample_big)
)
inv_diff_layers = [x .- y for (x, y) in zip(x0_layers, x0_layers_big)]
inv_err_layers = map(x -> map(norm, eachcol(x)), inv_diff_layers)

lpdfs_layer = intermediate_lpdfs(ts, q0, map(x -> Float32.(x), fwd_sample_big))
lpdfs_layer_big = intermediate_lpdfs(ts_big, q0_big, fwd_sample_big)
lpdfs_layer_big32 = Float32.(lpdfs_layer_big)

lpdfs_layer_diff = lpdfs_layer .- lpdfs_layer_big32
lpdfs_layer_diff_rel = abs.(lpdfs_layer_diff ./ lpdfs_layer_big32)

# exact density err is not small -- rough same magnitude as the sample inversion err
# but the relative lpdf err is smaller

#####################
# elbo err 
#####################
logp = Base.Fix1(logpdf, p)
# function logp_joint(z::AbstractVector{T}) where {T}
#     dim = div(length(z), 2)
#     x, ρ = z[1:dim], z[(dim + 1):end]
#     return logp(x) + logpdf(MvNormal(zeros(eltype(z), dim), I), ρ)
# end
function logp_joint(zs::AbstractMatrix{T}) where {T}
    dim = div(size(zs, 1), 2)
    xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]
    return logp(xs) + logpdf(MvNormal(zeros(eltype(zs), dim), I), ρs)
end

logp_big = Base.Fix1(logpdf, p_big)
function logp_joint_big(zs::AbstractMatrix{T}) where {T}
    dim = div(size(zs, 1), 2)
    xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]
    return logp_big(xs) + logpdf(MvNormal(zeros(eltype(zs), dim), I), ρs)
end

elbos = elbo_intermediate(ts, q0, logp_joint, Xs)
elbos_big = elbo_intermediate(ts_big, q0_big, logp_joint_big, Xs_big)

#####################
#  window computation
#####################
