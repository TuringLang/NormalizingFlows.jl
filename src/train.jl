function value_and_gradient! end

# zygote
function value_and_gradient!(
    at::ADTypes.AutoZygote, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(T))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, first(∇θ))
    return out
end

# ForwardDiff
# extract chunk size from AutoForwardDiff
getchunksize(::ADTypes.AutoForwardDiff{chunksize}) where {chunksize} = chunksize
function value_and_gradient!(
    at::ADTypes.AutoForwardDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    chunk_size = getchunksize(at)
    config = if isnothing(chunk_size)
        ForwardDiff.GradientConfig(f, θ)
    else
        ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(length(θ), chunk_size))
    end
    ForwardDiff.gradient!(out, f, λ, config)
    return out
end

# ReverseDiff without compiled tape
function value_and_gradient!(
    at::ADTypes.AutoReverseDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    tp = ReverseDiff.GradientTape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end

# Enzyme  
function value_and_gradient!(
    at::ADTypes.AutoEnzyme, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y = f(θ)
    DiffResults.value!(out, y)
    ∇θ = DiffResults.gradient(out)
    fill!(∇θ, zero(T))
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ))
    return out
end

function grad!(
    rng::AbstractRNG,
    at::ADTypes.AbstractADType,
    vo,
    θ_flat::AbstractVector{<:Real},
    reconstruct,
    out::DiffResults.MutableDiffResult,
    args...;
)
    # define opt loss function
    loss(θ_) = -vo(rng, reconstruct(θ_), args...)
    # compute loss value and gradient
    out = value_and_gradient!(at, loss, θ_flat, out)
    return out
end

#######################################################
# training loop for variational objectives that do not require input of data, e.g., reverse KL(elbo) without data subsampling in logp
#######################################################
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

function train(
    rng::AbstractRNG,
    at::ADTypes.AbstractADType,
    vo,
    θ₀::AbstractVector{T},
    re,
    args...;
    max_iters::Int=10000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    show_progress::Bool=true,
    callback=nothing,
) where {T<:Real}
    prog = ProgressMeter.Progress(
        max_iters; desc="Training", barlen=31, showspeed=true, enabled=show_progress
    )
    opt_stats = Vector{NamedTuple}(undef, max_iters)

    θ = copy(θ₀)
    diff_result = DiffResults.GradientResult(θ)
    # initialise optimiser state
    st = Optimisers.setup(optimiser, θ)

    time_elapsed = @elapsed for i in 1:max_iters
        grad!(rng, at, vo, θ, re, diff_result, args...)

        # save stats
        ls = DiffResults.value(diff_result)
        g = DiffResults.gradient(diff_result)
        stat_ = (iteration=i, loss=ls, gradient_norm=norm(g))
        opt_stats[i] = stat_

        # callback
        if !isnothing(callback)
            new_stat = callback(re, opt_stats, i)
            stat_ = !isnothing(new_stat) ? merge(new_stat, stat_) : stat_
        end

        # update optimiser state and parameters
        st, θ = Optimisers.update!(st, θ, DiffResults.gradient(diff_result))
        pm_next!(prog, stat_)
    end

    # return status of the optimiser for potential coninuation of training
    return θ, opt_stats, st
end