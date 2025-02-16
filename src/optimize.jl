#######################################################
# training loop for variational objectives 
#######################################################
function pm_next!(pm, stats::NamedTuple)
    return ProgressMeter.next!(pm; showvalues=[tuple(s...) for s in pairs(stats)])
end

_wrap_in_DI_context(args...) = DifferentiationInterface.Constant.([args...]) 

function _prepare_gradient(loss, adbackend, θ, args...)
    if isempty(args...)
        return DifferentiationInterface.prepare_gradient(loss, adbackend, θ)
    end
    return DifferentiationInterface.prepare_gradient(loss, adbackend, θ, _wrap_in_DI_context(args)...)
end

function _value_and_gradient(loss, prep, adbackend, θ, args...)
    if isempty(args...)
        return DifferentiationInterface.value_and_gradient(loss, prep, adbackend, θ)
    end
    return DifferentiationInterface.value_and_gradient(loss, prep, adbackend, θ, _wrap_in_DI_context(args)...)
end


"""
    optimize(
        ad::ADTypes.AbstractADType, 
        loss, 
        θ₀::AbstractVector{T}, 
        re, 
        args...; 
        kwargs...
    )

Iteratively updating the parameters `θ` of the normalizing flow `re(θ)` by calling `grad!`
 and using the given `optimiser` to compute the steps.

# Arguments
- `ad::ADTypes.AbstractADType`: automatic differentiation backend
- `loss`: a general loss function θ -> loss(θ, args...) returning a scalar loss value that will be minimised
- `θ₀::AbstractVector{T}`: initial parameters for the loss function (in the context of normalizing flows, it will be the flattened flow parameters)
- `re`: reconstruction function that maps the flattened parameters to the normalizing flow
- `args...`: additional arguments for `loss` (will be set as DifferentiationInterface.Constant)


# Keyword Arguments
- `max_iters::Int=10000`: maximum number of iterations
- `optimiser::Optimisers.AbstractRule=Optimisers.ADAM()`: optimiser to compute the steps
- `show_progress::Bool=true`: whether to show the progress bar. The default
  information printed in the progress bar is the iteration number, the loss value,
  and the gradient norm.
- `callback=nothing`: callback function with signature `cb(iter, opt_state, re, θ)`
  which returns a dictionary-like object of statistics to be displayed in the progress bar.
  re and θ are used for reconstructing the normalizing flow in case that user 
  want to further axamine the status of the flow.
- `hasconverged = (iter, opt_stats, re, θ, st) -> false`: function that checks whether the
  training has converged. The default is to always return false.
- `prog=ProgressMeter.Progress(
            max_iters; desc="Training", barlen=31, showspeed=true, enabled=show_progress
        )`: progress bar configuration

# Returns
- `θ`: trained parameters of the normalizing flow
- `opt_stats`: statistics of the optimiser
- `st`: optimiser state for potential continuation of training
"""
function optimize(
    adbackend,
    loss::Function, 
    θ₀::AbstractVector{<:Real}, 
    reconstruct::Function,
    args...;
    max_iters::Int=10000,
    optimiser::Optimisers.AbstractRule=Optimisers.ADAM(),
    show_progress::Bool=true,
    callback=nothing,
    hasconverged=(i, stats, re, θ, st) -> false,
    prog=ProgressMeter.Progress(
        max_iters; desc="Training", barlen=31, showspeed=true, enabled=show_progress
    ),
)
    time_elapsed = @elapsed begin 
        opt_stats = []

        # prepare loss and autograd
        θ = copy(θ₀)
        # grad = similar(θ)
        prep = _prepare_gradient(loss, adbackend, θ₀, args...)


        # initialise optimiser state
        st = Optimisers.setup(optimiser, θ)

        # general `hasconverged(...)` approach to allow early termination.
        converged = false
        i = 1
        while (i ≤ max_iters) && !converged
            # ls, g = DifferentiationInterface.value_and_gradient!(loss, grad, prep, adbackend, θ)
            ls, g = _value_and_gradient(loss, prep, adbackend, θ, args...)

            # Save stats
            stat = (iteration=i, loss=ls, gradient_norm=norm(g))

            # callback
            if !isnothing(callback)
                new_stat = callback(i, opt_stats, reconstruct, θ)
                stat = !isnothing(new_stat) ? merge(stat, new_stat) : stat
            end
            push!(opt_stats, stat)

            # update optimiser state and parameters
            st, θ = Optimisers.update!(st, θ, g)

            # check convergence
            i += 1
            converged = hasconverged(i, stat, reconstruct, θ, st)
            pm_next!(prog, stat)
        end
    end
    # return status of the optimiser for potential continuation of training
    return θ, map(identity, opt_stats), st, time_elapsed
end
