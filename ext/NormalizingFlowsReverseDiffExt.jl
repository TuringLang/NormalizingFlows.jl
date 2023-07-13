module NormalizingFlowsReverseDiffExt

if isdefined(Base, :get_extension)
    using NormalizingFlows
    using NormalizingFlows: AutoReverseDiff, MutableDiffResult
    using ReverseDiff
else
    using ..NormalizingFlows
    using ..NormalizingFlows: AutoReverseDiff, MutableDiffResult
    using ..ReverseDiff
end

# ReverseDiff without compiled tape
function NormalizingFlows.value_and_gradient!(
    ad::AutoReverseDiff, f, θ::AbstractVector{T}, out::MutableDiffResult
) where {T<:Real}
    tp = ReverseDiff.GradientTape(f, θ)
    ReverseDiff.gradient!(out, tp, θ)
    return out
end

end