module NormalizingFlowsReverseDiffExt

if isdefined(Base, :get_extension)
    using ADTypes
    using DiffResults
    using NormalizingFlows
    using ReverseDiff
else
    using ..ADTypes
    using ..DiffResults
    using ..NormalizingFlows
    using ..ReverseDiff
end

# ReverseDiff without compiled tape
function NormalizingFlows.value_and_gradient!(
    ad::ADTypes.AutoReverseDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    tp = ReverseDiff.GradientTape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end

end