using JLD2
include("setup.jl")
include("nuts.jl")

Random.seed!(123)
rng = Random.default_rng()

################################
# read target and contruct flow, flow_big, ts, its, its_big, its_big
# q0 and q0_big already defined in setup.jl
################################
res = JLD2.load("result/hamflow.jld2")
param_trained = res["param"]

ft = Float64
flow, ts, its, q0n, logp_new, r64 = set_precision_flow(ft, param_trained, q0)

setprecision(BigFloat, 1024)
bf = BigFloat
flow_big, ts_big, its_big, q0_big, logp_big = flow, ts, its, q0, logp
# flow_big, ts_big, its_big, q0_big, logp_big, re_big = set_precision_flow(
#     bf, param_trained, q0
# )

#####################
# test stability
######################

# forward sample stability
Xs = rand(q0, 20)
Xs_big = bf.(Xs)

# # check stability of big flow
# Xs_big .- its_big(ts_big(Xs_big))
diff = similar(Xs)
@threads for i in 1:size(Xs, 2)
    diff[:, i] = ft.(ts(Xs[:, i]) .- ts_big(Xs_big[:, i]))
end

diff = ts(Xs) .- ts_big(Xs_big)
dd = ft.(map(norm, eachcol(diff)))

fwd_sample = with_intermediate_results(ts, Xs)
fwd_sample_big = with_intermediate_results(ts_big, Xs_big)
fwd_sample_big32 = map(x -> ft.(x), fwd_sample_big)
fwd_diff_layer = fwd_sample .- fwd_sample_big
fwd_err_layer = reduce(
    hcat, map(x -> ft.(x), map(x -> map(norm, eachcol(x)), fwd_diff_layer))
)
####################
# fwd sample error scaling
####################
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
JLD2.save(
    "result/hamflow_fwd_err.jld2",
    "fwd_err_layer",
    fwd_err_layer,
    "fwd_sample",
    fwd_sample,
    "fwd_sample_big",
    fwd_sample_big32,
    "Xs",
    Xs,
    "s1",
    ft.(s1_layer_big),
    "s2",
    ft.(s2_layer_big),
    "s3",
    ft.(s3_layer_big),
    "s1_err",
    ft.(s1_layer_diff),
    "s2_err",
    ft.(s2_layer_diff),
    "s3_err",
    ft.(s3_layer_diff),
)

#####################
# density error
#####################
# Ys = ts(Xs)
μ = vec(
    [
        2.2038315273361597
        0.034873155889650446
        0.18405304373443232
        0.04980701527339299
        -0.30163078821700295
        0.7451950139534177
        -0.32560444859751797
        0.2894117116934399
        0.14156211903420354
    ],
)
nsample = 40
T = nuts(μ, 0.7, logp, ∇S, 20000 + 4000, 20000)[20001:end, :]
Ys = vcat(T'[:, 1:100:end], randn(ft, dims, nsample))
Ys_big = bf.(Ys)
diff_inv = its(Ys) .- its_big(Ys_big)
dd_inv = ft.(map(norm, eachcol(diff_inv)))

bwd_sample = with_intermediate_results(its, Ys)
# bwd_sample_big = map(x -> ft.(x), with_intermediate_results(its_big, Ys_big))
bwd_sample_big = with_intermediate_results(its_big, Ys_big)
bwd_diff_layer = bwd_sample .- bwd_sample_big
bwd_err_layer = ft.(reduce(hcat, map(x -> map(norm, eachcol(x)), bwd_diff_layer)))

err_lpdf = ft.(abs.(logpdf(flow, Ys) .- logpdf(flow_big, Ys_big)))

# test_seq = bwd_sample_big[end:-1:1]
# test_seq = fwd_sample_big
test_seq = [Ys for i in 1:length(bwd_sample)]
x0_layers = inverse_from_intermediate_layers(ts, map(x -> ft.(x), test_seq))
x0_layers_big = map(x -> ft.(x), inverse_from_intermediate_layers(ts_big, test_seq))
inv_diff_layers = [x .- y for (x, y) in zip(x0_layers, x0_layers_big)]
inv_err_layers = ft.(reduce(hcat, map(x -> map(norm, eachcol(x)), inv_diff_layers)))

lpdfs_layer = intermediate_lpdfs(ts, q0, map(x -> ft.(x), test_seq))
lpdfs_layer_big = intermediate_lpdfs(ts_big, q0_big, test_seq)
lpdfs_layer_big32 = ft.(lpdfs_layer_big)

lpdfs_layer_diff = lpdfs_layer .- lpdfs_layer_big32
lpdfs_layer_diff_rel = abs.(lpdfs_layer_diff ./ lpdfs_layer_big32)

# exact density err is not small -- rough same magnitude as the sample inversion err
# but the relative lpdf err is smaller
JLD2.save(
    "result/hamflow_bwd_err.jld2",
    "bwd_sample_big",
    ft.(bwd_sample_big),
    "bwd_sample",
    ft.(bwd_sample),
    "bwd_err_layer",
    ft.(bwd_err_layer),
    "inv_err_layer",
    ft.(inv_err_layers),
    "lpdfs_layer",
    ft.(lpdfs_layer),
    "lpdfs_layer_big",
    ft.(lpdfs_layer_big32),
    "lpdfs_layer_diff",
    ft.(lpdfs_layer_diff),
    "lpdfs_layer_diff_rel",
    ft.(lpdfs_layer_diff_rel),
)

#####################
# elbo err 
#####################

function logp_joint_big(zs::AbstractMatrix{T}) where {T}
    dim = div(size(zs, 1), 2)
    xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]
    return logp_big(xs) + logpdf(MvNormal(zeros(eltype(zs), dim), I), ρs)
end

elbos = elbo_intermediate(ts, q0, logp_joint, Xs)
elbos_big = elbo_intermediate(ts_big, q0_big, logp_joint_big, Xs_big)

JLD2.save("result/hamflow_elbo_err.jld2", "elbo", elbos, "elbo_big", elbos_big)

####################
# window computation
####################

# compute delta
delta_fwd = reduce(
    hcat, map(x -> map(norm, eachcol(x)), single_fwd_err(ts, fwd_sample_big, Xs))
)
delta_bwd = reduce(
    hcat, map(x -> map(norm, eachcol(x)), single_bwd_err(its, bwd_sample_big, Ys))
)
println(median(vec(delta_fwd)))
println(median(vec(delta_bwd)))
JLD2.save("result/hamflow_delta.jld2", "delta_fwd", delta_fwd, "delta_bwd", delta_bwd)

# compute window size
nsample = 40
δ = 1.0e-7
nlayers = length(fwd_sample)
window_fwd = zeros(nlayers, nsample)
window_bwd = zeros(nlayers, nsample)

using ProgressMeter
prog = ProgressMeter.Progress(nsample; desc="computing window", barlen=31, showspeed=true)
for i in 1:nsample
    x0 = rand(q0n)
    y0 = ts(x0)
    window_fwd[:, i] = all_shadowing_window(ts, x0, δ)
    window_bwd[:, i] = all_shadowing_window_inverse(its, y0, δ)
    ProgressMeter.next!(prog)
end

JLD2.save(
    "result/hamflow_shadowing.jld2",
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