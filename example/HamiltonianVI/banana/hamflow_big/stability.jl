using JLD2
include("setup.jl")

Random.seed!(123)
rng = Random.default_rng()

################################
# read target and contruct flow, flow_big, ts, its, its_big, its_big
# q0 and q0_big already defined in setup.jl
################################
res = JLD2.load("result/hamflow_big.jld2")
p = res["target"]
param_trained = res["param"]

# flow = re(param_trained)
# ts = flow.transform
# its = inverse(ts)
ft = Float32
flow, ts, its, q0n, r64 = set_precision_flow(ft, param_trained, q0)

setprecision(BigFloat, 2048)
bf = BigFloat
flow_big, ts_big, its_big, q0_big, re_big = set_precision_flow(bf, param_trained, q0)

compare_trained_and_untrained_flow_BN(flow, flow, p, 1000)
#####################
# test stability
######################

# forward sample stability
Xs = randn(ft, 2dims, 1000)
Xs_big = bf.(Xs)

# # check stability of big flow
# Xs_big .- its_big(ts_big(Xs_big))
diff = ts(Xs) .- ts_big(Xs_big)
dd = ft.(map(norm, eachcol(diff)))

fwd_sample = with_intermediate_results(ts, Xs)
fwd_sample_big = with_intermediate_results(ts_big, Xs_big)
fwd_sample_big32 = map(x -> ft.(x), fwd_sample_big)
fwd_diff_layer = fwd_sample .- fwd_sample_big
fwd_err_layer = reduce(
    hcat, map(x -> bf.(x), map(x -> map(norm, eachcol(x)), fwd_diff_layer))
)
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

s1_layer_diff = ft.(abs.(s1_layer .- s1_layer_big))
s2_layer_diff = ft.(abs.(s2_layer .- s2_layer_big))
s3_layer_diff = ft.(abs.(s3_layer .- s3_layer_big))
#= small fwd err in general: should see small window size =#
JLD2.save(
    "result/hamflow_big_fwd_err.jld2",
    "fwd_err_layer",
    fwd_err_layer,
    "fwd_sample",
    fwd_sample,
    "fwd_sample_big",
    fwd_sample_big32,
    "Xs",
    Xs,
    "s1",
    s1_layer_diff,
    "s2",
    s2_layer_diff,
    "s3",
    s3_layer_diff,
)

#####################
# density error
#####################
# Ys = ts(Xs)
Ys = vcat(rand(p, 100), randn(ft, 2, 100))
Ys_big = bf.(Ys)
diff_inv = its(Ys) .- its_big(Ys_big)
dd_inv = ft.(map(norm, eachcol(diff_inv)))

bwd_sample = with_intermediate_results(its, Ys)
# bwd_sample_big = map(x -> ft.(x), with_intermediate_results(its_big, Ys_big))
bwd_sample_big = with_intermediate_results(its_big, Ys_big)
bwd_diff_layer = bwd_sample .- bwd_sample_big
bwd_err_layer = ft.(reduce(hcat, map(x -> map(norm, eachcol(x)), bwd_diff_layer)))

# err_lpdf = ft.(abs.(logpdf(flow, Ys) .- logpdf(flow_big, Ys_big)))

# err_lpdf_rel =
#     ft.(
#         abs.(logpdf(flow_trained, Ys) .- logpdf(flow_big, Ys_big)) ./
#         abs.(logpdf(flow_big, Ys_big))
#     )

x0_layers = inverse_from_intermediate_layers(ts, map(x -> ft.(x), fwd_sample_big))
x0_layers_big = map(x -> ft.(x), inverse_from_intermediate_layers(ts_big, fwd_sample_big))
inv_diff_layers = [x .- y for (x, y) in zip(x0_layers, x0_layers_big)]
inv_err_layers = map(x -> map(norm, eachcol(x)), inv_diff_layers)

lpdfs_layer = intermediate_lpdfs(ts, q0, map(x -> ft.(x), fwd_sample_big))
lpdfs_layer_big = intermediate_lpdfs(ts_big, q0_big, fwd_sample_big)
lpdfs_layer_big32 = ft.(lpdfs_layer_big)

lpdfs_layer_diff = lpdfs_layer .- lpdfs_layer_big32
lpdfs_layer_diff_rel = abs.(lpdfs_layer_diff ./ lpdfs_layer_big32)

# exact density err is not small -- rough same magnitude as the sample inversion err
# but the relative lpdf err is smaller
JLD2.save(
    "result/hamflow_big_bwd_err.jld2",
    "bwd_err_layer",
    bwd_err_layer,
    "lpdfs_layer",
    lpdfs_layer,
    "lpdfs_layer_big",
    lpdfs_layer_big32,
    "lpdfs_layer_diff",
    lpdfs_layer_diff,
    "lpdfs_layer_diff_rel",
    lpdfs_layer_diff_rel,
)
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

JLD2.save("result/hamflow_big_elbo_err.jld2", "elbo", elbos, "elbo_big", elbos_big)
#####################
#  window computation
#####################

# compute delta
delta_fwd = reduce(
    hcat, map(x -> map(norm, eachcol(x)), single_fwd_err(ts, fwd_sample_big, Xs))
)
delta_bwd = reduce(
    hcat, map(x -> map(norm, eachcol(x)), single_bwd_err(its, bwd_sample_big, Ys))
)

# compute window size
nsample = 100
δ = 1.0e-6
nlayers = length(fwd_sample)
window_fwd = zeros(nlayers, nsample)
window_bwd = zeros(nlayers, nsample)

@threads for i in 1:nsample
    x0 = randn(4)
    y0 = ts(x0)
    window_fwd[:, i] = all_shadowing_window(ts, x0, δ)
    window_bwd[:, i] = all_shadowing_window(its, y0, δ)
end

JLD2.save(
    "result/hamflow_big_shadowing.jld2",
    "delta",
    δ,
    "window_fwd",
    window_fwd,
    "window_bwd",
    window_bwd,
    "delta_fwd",
    delta_fwd,
    "delta_bwd",
    delta_bwd,
)