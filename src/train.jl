"""
    value_and_gradient!(
        at::ADTypes.AbstractADType,
        f,
        θ::AbstractVector{T},
        out::DiffResults.MutableDiffResult
    ) where {T<:Real}

Compute the value and gradient of a function `f` at `θ` using the automatic
differentiation backend `at`.  The result is stored in `out`. 
The function `f` must return a scalar value. The gradient is stored in `out` as a
vector of the same length as `θ`.
"""
function value_and_gradient! end
# TODO: Make these definitions extensions to avoid loading unneecssary packages.
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
    ForwardDiff.gradient!(out, f, θ, config)
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

"""
    grad!(
        rng::AbstractRNG,
        at::ADTypes.AbstractADType,
        vo,
        θ_flat::AbstractVector{<:Real},
        reconstruct,
        out::DiffResults.MutableDiffResult,
        args...
    )

Compute the value and gradient for negation of the variational objective `vo` 
at `θ_flat` using the automatic differentiation backend `at`.  

Default implementation is provided for `at` where `at` is one of `AutoZygote`, 
`AutoForwardDiff`, `AutoReverseDiff` (with no compiled tape), and `AutoEnzyme`.
The result is stored in `out`.

# Arguments
- `rng::AbstractRNG`: random number generator
- `at::ADTypes.AbstractADType`: automatic differentiation backend
- `vo`: variational objective
- `θ_flat::AbstractVector{<:Real}`: flattened parameters of the normalizing flow
- `reconstruct`: function that reconstructs the normalizing flow from the flattened parameters
- `out::DiffResults.MutableDiffResult`: mutable diff result to store the value and gradient
- `args...`: additional arguments for `vo`
"""
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
# training loop for variational objectives that do not require input of data, 
# e.g., reverse KL(elbo) without data subsampling in logp
#######################################################
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

"""
    optimize(rng::AbstractRNG, at::ADTypes.AbstractADType, vo, θ₀::AbstractVector{T}, re, args...; kwargs...)

Iteratively updating the parameters `θ` of the normalizing flow `re(θ)` by calling `grad!`
 and using the given `optimiser` to compute the steps.

# Arguments
- `rng::AbstractRNG`: random number generator
- `at::ADTypes.AbstractADType`: automatic differentiation backend
- `vo`: variational objective
- `θ₀::AbstractVector{T}`: initial parameters of the normalizing flow
- `re`: function that reconstructs the normalizing flow from the flattened parameters
- `args...`: additional arguments for `vo`


# Keyword Arguments
- `max_iters::Int=10000`: maximum number of iterations
- `optimiser::Optimisers.AbstractRule=Optimisers.ADAM()`: optimiser to compute the steps
- `show_progress::Bool=true`: whether to show the progress bar. The default information printed in the progress bar is the iteration number, the loss value, and the gradient norm.
- `callback=nothing`: callback function with signature `cb(iter, opt_state, obj_val)`
  which returns a dictionary-like object of statistics to be displayed in the progress bar.
- `prog=ProgressMeter.Progress(max_iters; desc="Training", barlen=31, showspeed=true, enabled=show_progress)`: progress bar configuration

# Returns
- `θ`: trained parameters of the normalizing flow
- `opt_stats`: statistics of the optimiser
- `st`: optimiser state for potential continuation of training
"""
function optimize(
    rng::AbstractRNG,
    at::ADTypes.AbstractADType,
    vo,
    θ₀::AbstractVector{<:Real},
    re,
    args...;
    max_iters::Int=10000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    show_progress::Bool=true,
    callback=nothing,
    prog=ProgressMeter.Progress(
        max_iters; desc="Training", barlen=31, showspeed=true, enabled=show_progress
    ),
)
    opt_stats = Vector(undef, max_iters)

    θ = copy(θ₀)
    diff_result = DiffResults.GradientResult(θ)
    # initialise optimiser state
    st = Optimisers.setup(optimiser, θ)

    # TODO: Add support for general `hasconverged(...)` approach to allow early termination.
    time_elapsed = @elapsed for i in 1:max_iters
        grad!(rng, at, vo, θ, re, diff_result, args...)

        # save stats
        ls = DiffResults.value(diff_result)
        g = DiffResults.gradient(diff_result)
        stat = (iteration=i, loss=ls, gradient_norm=norm(g))
        opt_stats[i] = stat

        # callback
        if !isnothing(callback)
            new_stat = callback(re, opt_stats, i)
            stat = !isnothing(new_stat) ? merge(new_stat, stat) : stat
        end

        # update optimiser state and parameters
        st, θ = Optimisers.update!(st, θ, DiffResults.gradient(diff_result))
        pm_next!(prog, stat)
    end

    # return status of the optimiser for potential continuation of training
    return θ, map(identity, opt_stats), st
end
