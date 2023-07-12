module NormalizingFlowsReverseDiffExt

if isdefined(Base, :get_extension)
    using ReverseDiff
    using NormalizingFlows: value_and_gradient!
    using ADTypes
    using DiffResults
else
    using ..ReverseDiff
    using ..NormalizingFlows: value_and_gradient!
    using ..ADTypes
    using ..DiffResults
end

# ReverseDiff without compiled tape
function value_and_gradient!(
    ad::ADTypes.AutoReverseDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    tp = ReverseDiff.GradientTape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end
end