using Random, Distributions, LinearAlgebra, Bijectors

# accessing the trained flow by looking at the first 2 dimensions
function compare_trained_and_untrained_flow_BN(
    flow_trained::Bijectors.MultivariateTransformed,
    flow_untrained::Bijectors.MultivariateTransformed,
    true_dist::ContinuousMultivariateDistribution,
    n_samples::Int;
    kwargs...,
)
    samples_trained = rand_batch(flow_trained, n_samples)
    samples_untrained = rand_batch(flow_untrained, n_samples)
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

#######################
# training function for InvertibleNetworks
########################

function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function train_invertible_networks!(G, loss, data_loader, n_epoch, opt)
    max_iters = n_epoch * length(data_loader)
    prog = ProgressMeter.Progress(
        max_iters; desc="Training", barlen=31, showspeed=true, enabled=true
    )

    loss_l2_list = []
    loss_lgdet_list = []
    nnls = []

    # training loop
    time_elapsed = @elapsed for (i, xs) in enumerate(IterTools.ncycle(data_loader, n_epoch))
        losses = loss(G, xs) #sets gradients of G

        push!(loss_l2_list, losses[1])
        push!(loss_lgdet_list, losses[2])

        nnl = losses[1] - losses[2]
        push!(nnls, nnl)

        grad_norm = 0
        for p in get_params(G)
            grad_norm += sum(abs2, p.grad)
            Flux.update!(opt, p.data, p.grad)
        end
        grad_norm = sqrt(grad_norm)

        stat = (iteration=i, neg_log_llh=nnl, gradient_norm=grad_norm)
        pm_next!(prog, stat)
    end
    return nnls, loss_l2_list, loss_lgdet_list
end
