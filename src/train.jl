using Distributions, LinearAlgebra
using Optimisers
using ProgressMeter
using Random
using ADTypes

using Zygote, ForwardDiff, ReverseDiff, Enzyme

function value_and_gradient end

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
    at::ADTypes.AbstractADType,
    vo,
    θ_flat::AbstractVector{<:Real},
    reconstruct,
    out::DiffResults.MutableDiffResult,
    args...;
    rng::AbstractRNG=Random.GLOBAL_RNG,
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
function train!(
    at::ADTypes.AbstractADType,
    vo,
    θ₀::AbstractVector{T},
    re,
    args...;
    max_iters::Int=10000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    rng::AbstractRNG=Random.GLOBAL_RNG,
) where {T<:Real}

    # progress bar
    prog = ProgressMeter.Progress(max_iters, 1, "Training...", 0)

    θ = copy(θ₀)
    diff_result = DiffResults.GradientResult(θ)

    # initialise optimiser state
    st = Optimisers.setup(optimiser, θ)

    losses = zeros(max_iters)
    time_elapsed = @elapsed for i in 1:max_iters
        grad!(at, vo, θ, re, diff_result, args...; rng=rng)
        losses[i] = DiffResults.value(diff_result)

        # update optimiser state and parameters
        st, θ = Optimisers.update!(st, θ, DiffResults.gradient(diff_result))
        ProgressMeter.next!(prog)
    end

    # return status of the optimiser for potential coninuation of training
    return losses, θ, st
end

# training loop for variational objectives that require input of data, e.g., forward KL(MLE), elbo with data subsampling in logp
function train_pass_in_data!() end
