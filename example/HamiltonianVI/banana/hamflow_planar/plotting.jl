using Plots, JLD2, StatsPlots
using StatsBase

include("../../../util.jl")

##########################
# read result
##########################
res_fwd = JLD2.load("result/hamflow_planar_fwd_err.jld2")
res_bwd = JLD2.load("result/hamflow_planar_bwd_err.jld2")
res_elbo = JLD2.load("result/hamflow_planar_elbo_err.jld2")
# res_llh = JLD2.load("result/MLP_llh_err.jld2")
res_shadowing = JLD2.load("result/hamflow_planar_shadowing.jld2")

##########################3
# sample pass err
###########################
fwd_err_layer = res_fwd["fwd_err_layer"]
bwd_err_layer = res_bwd["bwd_err_layer"]
nlayers = size(fwd_err_layer, 2)
plot(
    1:nlayers,
    vec(median(fwd_err_layer; dims=1));
    ribbon=get_percentiles(fwd_err_layer; byrow=false),
    lw=3,
    label="Fwd",
    xlabel="#transformations",
    ylabel="Error",
    title="NF numerical error",
    legend=:bottomright,
)
plot!(
    1:nlayers,
    vec(median(bwd_err_layer; dims=1));
    ribbon=get_percentiles(bwd_err_layer; byrow=false),
    lw=3,
    label="Bwd",
)
plot!(;
    yaxis=:log10,
    size=(800, 500),
    xrotation=0,
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
)
savefig("figure/flow_err_log.png")

#################################3
# lpdfs
##################################
lpdfs_layer_diff = abs.(res_bwd["lpdfs_layer_diff"]) .+ 1e-8
lpdfs_layer_diff_rel = abs.(res_bwd["lpdfs_layer_diff_rel"]) .+ 1e-10
nlayers = size(lpdfs_layer_diff, 2)
plot(
    1:nlayers,
    vec(median(lpdfs_layer_diff; dims=1));
    ribbon=get_percentiles(lpdfs_layer_diff; byrow=false),
    lw=3,
    label="",
    xlabel="#transformations",
    ylabel="error",
    title="NF log-density error",
)
plot!(;
    yaxis=:log10,
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
)
savefig("figure/lpdf_err.png")

plot(
    1:nlayers,
    vec(median(lpdfs_layer_diff_rel; dims=1));
    ribbon=get_percentiles(lpdfs_layer_diff_rel; byrow=false),
    lw=3,
    label="",
    xlabel="#transformations",
    ylabel="error",
    title="NF log-density error",
)
plot!(;
    yaxis=:log10,
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
)
savefig("figure/lpdf_err_rel.png")

# ###################3
# # elbo
# #####################
elbos = res_elbo["elbo"]
elbos_big = Float32.(res_elbo["elbo_big"])

plot(
    1:nlayers,
    elbos;
    lw=3,
    label="numerical",
    xlabel="#transformations",
    ylabel="ELBO",
    title="NF ELBO est.",
)
plot!(1:nlayers, elbos_big; lw=3, label="exact")
plot!(;
    # yaxis=:log10,
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
)
savefig("figure/elbo.png")

# ###################3
# shadowing
# #####################

delta = res_shadowing["delta"]
window_fwd = res_shadowing["window_fwd"]
window_bwd = res_shadowing["window_bwd"]
delta_fwd = res_shadowing["delta_fwd"] .+ 1e-8
delta_bwd = res_shadowing["delta_bwd"] .+ 1e-8

p1 = boxplot(
    ["Fwd err." "Bwd err."],
    [vec(delta_fwd) vec(delta_bwd)];
    legend=false,
    title="NF single map err.",
)
plot!(p1; xlabel="", ylabel="Error", yaxis=:log10)
plot!(;
    size=(800, 500),
    # yticks=[1e-3, 1e-6, 1e-10],
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
)
savefig(p1, joinpath("figure/", "delta.png"))

p1 = plot(
    1:nlayers,
    vec(median(window_fwd; dims=2));
    ribbon=get_percentiles(window_fwd),
    lw=3,
    label="Fwd",
    xlabel="#transformations",
    ylabel="",
)
p2 = plot(
    1:nlayers,
    vec(median(window_bwd; dims=2));
    ribbon=get_percentiles(window_bwd),
    lw=3,
    label="Bwd",
    xlabel="#transformations",
    ylabel="",
)
# put them side by side
pp = plot(p1, p2; layout=(1, 2), title="NF window size")
plot!(;
    size=(1200, 600),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
)

savefig(pp, "figure/window.png")

p1 = plot(
    1:nlayers,
    vec(median(window_fwd; dims=2));
    ribbon=get_percentiles(window_fwd),
    lw=3,
    label="Fwd",
    xlabel="#transformations",
    ylabel="",
    yaxis=:log10,
)
p2 = plot(
    1:nlayers,
    vec(median(window_bwd; dims=2));
    ribbon=get_percentiles(window_bwd),
    lw=3,
    label="Bwd",
    xlabel="#transformations",
    ylabel="",
    yaxis=:log10,
)
# put them side by side
pp = plot(p1, p2; layout=(1, 2), title="NF window size")
plot!(;
    size=(1200, 600),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
)

savefig(pp, "figure/window_log.png")