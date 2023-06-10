using Distributions, LinearAlgebra
using Flux, Zygote
using Optimisers
using ProgressMeter
using Random
import AbstractDifferentiation as AD

# TODO:
# support difference AD systems 
# now by default using Zygote for the AD backend

# Question: maybe its easier to use AbstractDifferentiation. 
# This allows us to switch between different AD backend easily instead of implementing 5 different `grad!` function for different AD backend` 

# compute grad and loss using Zygote 
function grad!(
    ab::AD.AbstractBackend,
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
    ls, ∇θ = AD.value_and_gradient(ab, loss, θ_flat)

    DiffResults.value!(out, ls)
    DiffResults.gradient!(out, first(∇θ))
    return out
end

# training loop for variational objectives that do not require input of data, e.g., reverse KL(elbo)
function train!(
    ab::AD.AbstractBackend,
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
        grad!(ab, vo, θ, re, diff_result, args...; rng=rng)
        losses[i] = DiffResults.value(diff_result)

        # update optimiser state and parameters
        st, θ = Optimisers.update!(st, θ, DiffResults.gradient(diff_result))
        ProgressMeter.next!(prog)
    end

    # return status of the optimiser for potential coninuation of training
    return losses, θ, st
end

# training loop for variational objectives that require input of data, e.g., forward KL(MLE)
function train_pass_in_data!() end
