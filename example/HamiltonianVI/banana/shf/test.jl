using JLD2, ProgressMeter
using PlotlyJS, Plots
using LaTeXStrings
const pjs = PlotlyJS
include("setup.jl")

Random.seed!(123)
rng = Random.default_rng()

################################
# read target and contruct flow, flow_big, ts, its, its_big, its_big
# q0 and q0_big already defined in setup.jl
################################
res = JLD2.load("result/hamflow.jld2")
p = res["target"]
param_trained = res["param"]

# flow = re(param_trained)
# ts = flow.transform
# its = inverse(ts)
ft = Float64
flow, ts, its, q0n, r64 = set_precision_flow(ft, param_trained, q0)

setprecision(BigFloat, 2048)
bf = BigFloat
flow_big, ts_big, its_big, q0_big, re_big = set_precision_flow(bf, param_trained, q0)

pp = check_trained_flow(
    flow,
    p,
    1000;
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
    legend=:bottom,
)
savefig(pp, "figure/trained_flow.png")

# ########################
# # lpdf vis
# ######################
X = [-20.001:0.1:20;]
Y = [-20.001:0.1:15;]
Ds = zeros(length(X), length(Y))
Dd = zeros(length(X), length(Y))

# lpdf_est, Error
n1, n2 = size(X, 1), size(Y, 1)
@showprogress for i in 1:n1
    grid = reduce(hcat, [[X[i], y, 0.0, 0.0] for y in Y])
    Ds[i, :] = logpdf(flow, grid)
    Dd[i, :] = logp(grid[1:2, :])
end

# @showprogress for i in 1:n1
#     @threads for j in 1:n2
#         Ds[i, j] = logpdf(flow, [X[i], Y[j], 0.0, 0.0])
#         Dd[i, j] = logp([X[i], Y[j]])
#     end
# end

JLD2.save("result/lpdfs_vis.jld2", "lpdfs", Ds, "true", Dd)

res = JLD2.load("result/lpdfs_vis.jld2")
Ds = res["lpdfs"]
Dd = res["true"]

layout = pjs.Layout(;
    width=500,
    height=500,
    scene=pjs.attr(;
        xaxis=pjs.attr(; showticklabels=true, visible=true),
        yaxis=pjs.attr(; showticklabels=true, visible=true),
        zaxis=pjs.attr(; showticklabels=true, visible=true),
    ),
    margin=pjs.attr(; l=0, r=0, b=0, t=0, pad=0),
    colorscale="Vird",
)
p_est = pjs.plot(pjs.surface(; z=Ds, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_est, joinpath("figure/", "lpdf_est.png"))

p_target = pjs.plot(pjs.surface(; z=Dd, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_target, joinpath("figure/", "lpdf.png"))

###################################
#compare L_q, L_p
###################################
function ∇logp_joint(zs::AbstractMatrix{T}) where {T}
    dim = div(size(zs, 1), 2)
    xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]
    Gs = similar(zs)
    Gs[1:dim, :] .= ∇S(xs)
    Gs[(dim + 1):end, :] .= -ρs
    return Gs
end
logq = Base.Fix1(logpdf, flow)
∇logq_joint(x) = Zygote.gradient(logq, x)[1]

Ys = vcat(rand(p, 100), randn(ft, 2, 100))

Gp = ∇logp_joint(Ys)
Gq = reduce(hcat, map(∇logq_joint, eachcol(Ys)))
Lp = map(norm, eachcol(Gp))
Lq = map(norm, eachcol(Gq))

JLD2.save("result/Lip.jld2", "Lp", Lp, "Lq", Lq, "Ys", Ys, "Gp", Gp, "Gq", Gq)

res = JLD2.load("result/Lip.jld2")
p1 = boxplot(
    [L"$||\nabla\log p ||$" L"$||\nabla \log q||$"],
    [res["Lp"] res["Lq"]];
    legend=false,
    yaxis=:log10,
)
plot!(;
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
    title="Comp. of local Lip. const.",
)
savefig(p1, "figure/LipConst_last_layer.png")

# scaling of Lq over layers
Lq_layers = intermediate_Lqs(ts, q0, Ys)

# JLD2.save("result/LipLayer.jld2", "Lq", Lq_layers, "Ys", Ys)

Lq_layers = JLD2.load("result/LipLayer.jld2")["Lq"]
p1 = Plots.plot(
    1:size(Lq_layers, 2),
    vec(median(Lq_layers; dims=1));
    ribbon=get_percentiles(Lq_layers; byrow=false),
    lw=3,
    label="",
    xlabel="#transformations",
    ylabel="",
    yaxis=:log10,
)
Plots.plot!(;
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
    title=L"HamFlow $||\nabla \log q||$",
    # title=L"HamFlow lower bound of $L_{p, ϵ}$",
)
Plots.savefig(p1, "figure/Lq_scaling.png")

# scaling of Lp over layers
Xs = rand(q0, 1000)
fwd_sample = with_intermediate_results(ts, Xs)
# function norm_∇logp_joint(xs::AbstractMatrix)
#     g1 = ∇S(@view(xs[1:2, :]))
#     g2 = -@view(xs[3:4, :])
#     return vec(sqrt.(sum(abs2, g1; dims = 1).+ sum(abs2, g2; dims = 1)))
# end
function norm_∇logp_joint(x::AbstractVector)
    g1 = ∇S(@view(x[1:2]))
    g2 = -@view(x[3:4])
    return sqrt(sum(abs2, g1) + sum(abs2, g2))
end
function local_smooth(p, x, ϵ)
    b, var = p.b, p.var
    x1, x2, x3, x4 = x
    m11 = abs(6b^2 * x1^2 + 2b * x2 - 2var * b^2 + 1 / var)
    m1 = max(m11, 1)
    m21 = 12b^2 * abs(x1) + 12b^2(ϵ + ϵ^2)
    m2 = max(m21, 2b * ϵ)
    return 2b * abs(x1) + m1 + m2
end
function local_lip(p, x::AbstractVector, ϵ::Real)
    return norm_∇logp_joint(x) + local_smooth(p, x, ϵ) * ϵ
end
function local_lip(p, xs::AbstractMatrix, ϵ::Real)
    return map(x -> local_lip(p, x, ϵ), eachcol(xs))
end

function local_lip_scaling(p, fwd_sample, ϵs)
    nlayers = length(fwd_sample)
    Lps = zeros(nlayers)
    @threads for i in 1:nlayers
        xs = fwd_sample[i]
        Lps[i] = mean(local_lip(p, xs, ϵs[i]))
    end
    return Lps
end

res = JLD2.load("result/hamflow_shadowing.jld2")
window_fwd = res["window_fwd"]
ϵs = vec(maximum(window_fwd; dims=2))
Lp_layer = local_lip_scaling(p, fwd_sample, ϵs)

JLD2.save(
    "result/LipLayer.jld2",
    "Lq",
    Lq_layers,
    "Ys",
    Ys,
    "fwd_sample",
    fwd_sample,
    "Lp",
    Lp_layer,
    "ϵs",
    ϵs,
)

Lp_layer = JLD2.load("result/LipLayer.jld2")["Lp"]
p1 = Plots.plot(
    1:size(Lp_layer, 1), Lp_layer; lw=3, label="", xlabel="#transformations", ylabel=""
)
Plots.plot!(;
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
    title=L"HamFlow upper bound of $L_{p, ϵ}$",
)
Plots.savefig(p1, "figure/Lp_scaling.png")

# 
# el_err = abs.(elbos .- elbos_big)
# ϵs = res["ϵs"]
# Lp_layers = res["Lp"]

# bdd = ϵs .* (Lp_layer .+ 2) .+ ϵs .^ 2
# plot(el_err; lw=3, label="el")
# plot!(bdd; lw=3)