using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using FunctionChains
using NormalizingFlows

using Plots
Random.seed!(123)
rng = Random.default_rng()

include("targets/banana.jl")
######################################
# learn the target using planar flow 
######################################
function create_planar_flow(n_layers::Int, q₀)
    d = length(q₀)
    Ls = [
        PlanarLayer(randn(Float32, d), randn(Float32, d), randn(Float32, 1)) for
        _ in 1:n_layers
    ]
    ts = fchain(Ls)
    return transformed(q₀, ts)
end

# create target p
p = Banana(2, 1.0f-1)
logp = Base.Fix1(logpdf, p)

# create a 10-layer planar flow
flow = create_planar_flow(20, MvNormal(zeros(Float32, 2), I))
flow_untrained = deepcopy(flow)

# train the flow
sample_per_iter = 1
cb(re, opt_stats, i) = (sample_per_iter=sample_per_iter,)
flow_trained, stats, _ = NF(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters=100_000,
    optimiser=Optimisers.ADAM(),
    callback=cb,
)
losses = map(x -> x.loss, stats)

######################################
# evaluate the trained flow
######################################

function compare_trained_and_untrained_flow(
    flow_trained, flow_untrained, true_dist, n_samples
)
    samples_trained = rand(flow_trained, n_samples)
    samples_untrained = rand(flow_untrained, n_samples)
    samples_true = rand(true_dist, n_samples)

    p = scatter(
        samples_true[1, :],
        samples_true[2, :];
        label="True Distribution",
        color=:blue,
        markersize=2,
        alpha=0.5,
    )
    scatter!(
        p,
        samples_untrained[1, :],
        samples_untrained[2, :];
        label="Untrained Flow",
        color=:red,
        markersize=2,
        alpha=0.5,
    )
    scatter!(
        p,
        samples_trained[1, :],
        samples_trained[2, :];
        label="Trained Flow",
        color=:green,
        markersize=2,
        alpha=0.5,
    )

    xlabel!(p, "X")
    ylabel!(p, "Y")
    title!(p, "Comparison of Trained and Untrained Flow")

    return p
end

plot(losses; label="Loss", linewidth=2)
compare_trained_and_untrained_flow(flow_trained, flow_untrained, p, 1000)
