using JLD2
include("setup.jl")

Random.seed!(123)
rng = Random.default_rng()

################################
# read target and contruct flow, flow_big, ts, its, its_big, its_big
# q0 and q0_big already defined in setup.jl
################################
res = JLD2.load("result/hamflow_planar.jld2")
p = res["target"]
param_trained = res["param"]

flow = re(param_trained)
ts = flow.transform
its = inverse(ts)

setprecision(BigFloat, 2048)
bf = BigFloat
flow_big, ts_big, its_big, q0_big, re_big = set_precision_flow(bf, param_trained, q0)

#####################
# test stability
######################

# forward sample stability
Xs = randn(Float32, 2dims, 1000)
Xs_big = ft.(Xs)

# # check stability of big flow
# Xs_big .- its_big(ts_big(Xs_big))
diff = ts(Xs) .- ts_big(Xs_big)
dd = Float32.(map(norm, eachcol(diff)))

fwd_sample = with_intermediate_results(ts, Xs)
fwd_sample_big = with_intermediate_results(ts_big, Xs_big)
fwd_sample_big32 = map(x -> Float32.(x), fwd_sample_big)
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

# compute delta
delta_fwd = single_fwd_err(ts, fwd_sample_big, Xs)
delta_bwd = single_bwd_err(its, bwd_sample_big, Ys)

# compute window size
nsample = 100
δ = 1.0f-7
nlayers = length(fwd_sample)
ϵs_fwd = zeros(Float32, nlayers, nsample)
ϵs_bwd = zeros(Float32, nlayers, nsample)

@threads for i in 1:nsample
    fwd_trjs = vcat(Xs[:, i], [fwd_sample[j][:, i] for j in 1:(nlayers - 1)])
    bwd_trjs = vcat(Ys[:, i], [bwd_sample[j][:, i] for j in 1:(nlayers - 1)])

    Ms_fwd = flow_fwd_jacobians(ts, fwd_trjs)
    Ms_bwd = flow_bwd_jacobians(its, bwd_trjs)

    ϵs_fwd[:, i] = all_shadowing_window(Ms_fwd, δ)
    ϵs_bwd[:, i] = all_shadowing_window(Ms_bwd, δ)
end
