using JLD2, ProgressMeter
using PlotlyJS, Plots
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

# pp = check_trained_flow(flow, p, 1000)
# savefig(pp, "figure/trained_flow.png")

# ########################
# # lpdf vis
# ######################
# X = [-20.001:0.1:20;]
# Y = [-20.001:0.1:15;]
# Ds = zeros(length(X), length(Y))
# Dd = zeros(length(X), length(Y))

# # lpdf_est, Error
# n1, n2 = size(X, 1), size(Y, 1)
# @showprogress for i in 1:n1
#     grid = reduce(hcat, [[X[i], y, 0.0, 0.0] for y in Y])
#     Ds[i, :] = logpdf(flow, grid)
#     Dd[i, :] = logp(grid[1:2, :])
# end

# # @showprogress for i in 1:n1
# #     @threads for j in 1:n2
# #         Ds[i, j] = logpdf(flow, [X[i], Y[j], 0.0, 0.0])
# #         Dd[i, j] = logp([X[i], Y[j]])
# #     end
# # end

# JLD2.save("result/lpdfs_vis.jld2", "lpdfs", Ds, "true", Dd)

# res = JLD2.load("result/lpdfs_vis.jld2")
# Ds = res["lpdfs"]
# Dd = res["true"]

# layout = pjs.Layout(;
#     width=500,
#     height=500,
#     scene=pjs.attr(;
#         xaxis=pjs.attr(; showticklabels=true, visible=true),
#         yaxis=pjs.attr(; showticklabels=true, visible=true),
#         zaxis=pjs.attr(; showticklabels=true, visible=true),
#     ),
#     margin=pjs.attr(; l=0, r=0, b=0, t=0, pad=0),
#     colorscale="Vird",
# )
# p_est = pjs.plot(pjs.surface(; z=Ds, x=X, y=Y, showscale=false), layout)
# pjs.savefig(p_est, joinpath("figure/", "lpdf_est.png"))

# p_target = pjs.plot(pjs.surface(; z=Dd, x=X, y=Y, showscale=false), layout)
# pjs.savefig(p_target, joinpath("figure/", "lpdf.png"))

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

Ys = vcat(rand(p, 200), randn(ft, 2, 200))

Gp = ∇logp_joint(Ys)
Gq = map(∇logq_joint, eachcol(Ys))
Lp = map(norm, eachcol(Gp))
Lp = map(norm, eachcol(Gq))

JLD2.save("result/Lip.jld2", "Lp", Lp, "Lq", Lq, "Ys", Ys, "Gp", Gp, "Gq", Gq)
