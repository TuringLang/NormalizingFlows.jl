using Random, Distributions, LinearAlgebra, Bijectors

# accessing the trained flow by looking at the first 2 dimensions
function compare_trained_and_untrained_flow(
    flow_trained::Bijectors.MultivariateTransformed,
    flow_untrained::Bijectors.MultivariateTransformed,
    true_dist::ContinuousMultivariateDistribution,
    n_samples::Int;
    kwargs...,
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
    plot!(; kwargs...)

    xlabel!(p, "X")
    ylabel!(p, "Y")
    title!(p, "Comparison of Trained and Untrained Flow")

    return p
end

function create_flow(Ls, q₀)
    ts = fchain(Ls)
    return transformed(q₀, ts)
end