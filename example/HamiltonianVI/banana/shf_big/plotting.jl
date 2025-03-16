using Plots, JLD2, StatsPlots
using StatsBase
using Random
using FunctionChains
ft = Float64
include("../../../util.jl")

##########################
# read result
##########################
res_fwd = JLD2.load("result/hamflow_fwd_err.jld2")
res_bwd = JLD2.load("result/hamflow_bwd_err.jld2")
res_elbo = JLD2.load("result/hamflow_elbo_err.jld2")
res_shadowing = JLD2.load("result/hamflow_shadowing.jld2")
res_delta = JLD2.load("result/hamflow_delta.jld2")

function ribbon_plot(x, y; byrow=false, kwargs...)
    pp = plot(
        x,
        vec(median(y; dims=1));
        ribbon=get_percentiles(y; byrow=byrow),
        lw=3,
        size=(800, 500),
        xrotation=0,
        xtickfontsize=30,
        ytickfontsize=30,
        margin=10Plots.mm,
        guidefontsize=30,
        titlefontsize=30,
        kwargs...,
    )
    return pp
end

##########################3
# sample pass err
###########################
fwd_err_layer = res_fwd["fwd_err_layer"]
bwd_err_layer = res_bwd["inv_err_layer"]
nlayers = size(fwd_err_layer, 2)
plot(
    1:nlayers,
    vec(median(fwd_err_layer; dims=1));
    ribbon=get_percentiles(fwd_err_layer; byrow=false),
    lw=3,
    label="Fwd",
    xlabel="#transformations",
    ylabel="Error",
    title="HamFlow numerical error",
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

plot(
    1:nlayers,
    vec(median(fwd_err_layer; dims=1));
    ribbon=get_percentiles(fwd_err_layer; byrow=false),
    lw=3,
    label="Fwd",
    xlabel="#transformations",
    ylabel="Error",
    title="HamFlow numerical error",
    legend=:topleft,
)
plot!(
    1:nlayers,
    vec(median(bwd_err_layer; dims=1));
    ribbon=get_percentiles(bwd_err_layer; byrow=false),
    lw=3,
    label="Bwd",
)
plot!(;
    size=(800, 500),
    xrotation=0,
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
)
savefig("figure/flow_err.png")

s1_layer = ft.(res_fwd["s1"])
s2_layer = ft.(res_fwd["s2"])
s3_layer = ft.(res_fwd["s3"])

s1_layer_diff = res_fwd["s1_err"]
s2_layer_diff = res_fwd["s2_err"]
s3_layer_diff = res_fwd["s3_err"]

plot(
    1:nlayers,
    [s1_layer_diff ./ s1_layer s2_layer_diff ./ s2_layer s3_layer_diff ./ s3_layer];
    lw=3,
    label=["|x|" "sin(x)+1" "sigmoid"],
    xlabel="#transformations",
    ylabel="Rel. err.",
    title="HamFlow sampling error",
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
    legend=:bottomright,
    # yticks=[1e-15, 1e-10, 1e-5, 1e-1],
)
savefig("figure/stat_err_log.png")

plot(
    1:nlayers,
    [s1_layer_diff ./ s1_layer s2_layer_diff ./ s2_layer s3_layer_diff ./ s3_layer];
    lw=3,
    label=["|x|" "sin(x)+1" "sigmoid"],
    xlabel="#transformations",
    ylabel="Rel. err.",
    title="HamFlow sampling error",
)
plot!(;
    size=(800, 500),
    xrotation=0,
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
    legend=:topleft,
)
savefig("figure/stat_err.png")

#################################3
# lpdfs
##################################
lpdfs_layer_diff = abs.(res_bwd["lpdfs_layer_diff"])
lpdfs_layer_diff_rel = abs.(res_bwd["lpdfs_layer_diff_rel"])
nlayers = size(lpdfs_layer_diff, 2)
plot(
    1:nlayers,
    vec(median(lpdfs_layer_diff; dims=1));
    ribbon=get_percentiles(lpdfs_layer_diff; byrow=false),
    lw=3,
    label="",
    xlabel="#transformations",
    ylabel="error",
    title="HamFlow log-density error",
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
savefig("figure/lpdf_err_log.png")

plot(
    1:nlayers,
    vec(median(lpdfs_layer_diff; dims=1));
    ribbon=get_percentiles(lpdfs_layer_diff; byrow=false),
    lw=3,
    label="",
    xlabel="#transformations",
    ylabel="error",
    title="HamFlow log-density error",
)
plot!(;
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
    ylabel="Rel. error",
    title="HamFlow log-density error",
)
plot!(;
    yaxis=:log10,
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
    # yticks=[1e-15, 1e-10, 1e-5, 1e-1],
)
savefig("figure/lpdf_err_rel_log.png")

plot(
    1:nlayers,
    vec(median(lpdfs_layer_diff_rel; dims=1));
    ribbon=get_percentiles(lpdfs_layer_diff_rel; byrow=false),
    lw=3,
    label="",
    xlabel="#transformations",
    ylabel="Rel. error",
    title="HamFlow log-density error",
)
plot!(;
    size=(800, 500),
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    titlefontsize=30,
)
savefig("figure/lpdf_err_rel.png")

#####################
# elbo
#####################
elbos = res_elbo["elbo"]
elbos_big = Float64.(res_elbo["elbo_big"])

plot(
    1:nlayers,
    elbos;
    lw=3,
    label="numerical",
    xlabel="#transformations",
    ylabel="ELBO",
    title="HamFlow ELBO est.",
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

# #####################3
# # shadowing
# ######################

delta = res_shadowing["delta"]
window_fwd = res_shadowing["window_fwd"]
window_bwd = res_shadowing["window_bwd"]
delta_fwd = res_shadowing["delta_fwd"]
delta_bwd = res_shadowing["delta_bwd"]

p1 = boxplot(
    ["Fwd err." "Bwd err."],
    [vec(delta_fwd) vec(delta_bwd)];
    legend=false,
    title="HamFlow single map err.",
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
plot!(
    1:nlayers,
    vec(median(window_bwd; dims=2));
    ribbon=get_percentiles(window_bwd),
    lw=3,
    label="Bwd",
    xlabel="#transformations",
    ylabel="",
)
plot!(;
    title="HamFlow window size",
    size=(800, 500),
    # yticks=[1e-3, 1e-6, 1e-10],
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
)

savefig(p1, "figure/window.png")

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
plot!(
    1:nlayers,
    vec(median(window_bwd; dims=2));
    ribbon=get_percentiles(window_bwd),
    lw=3,
    label="Bwd",
    xlabel="#transformations",
    ylabel="",
    yaxis=:log10,
)
plot!(;
    title="HamFlow window size",
    size=(800, 500),
    # yticks=[1e-3, 1e-6, 1e-10],
    xtickfontsize=30,
    ytickfontsize=30,
    margin=10Plots.mm,
    guidefontsize=30,
    legendfontsize=20,
    titlefontsize=30,
    legend=:bottomright,
)

savefig(p1, "figure/window_log.png")