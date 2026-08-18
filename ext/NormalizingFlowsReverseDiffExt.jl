module NormalizingFlowsReverseDiffExt

using NormalizingFlows
using ReverseDiff: ReverseDiff, TrackedMatrix, value

# The bin index, the inside mask, and the root sign mask are piecewise constant, so no
# derivative flows through them. Evaluating them on primal values avoids a crash in
# ReverseDiff's array broadcast, which handles neither the comparison nor `ifelse` when only
# some arguments are tracked.
function NormalizingFlows._rqs_bin(knots::ReverseDiff.TrackedArray, x::AbstractMatrix)
    return NormalizingFlows._rqs_bin(value(knots), x)
end
function NormalizingFlows._rqs_bin(knots::AbstractArray, x::TrackedMatrix)
    return NormalizingFlows._rqs_bin(knots, value(x))
end
function NormalizingFlows._rqs_bin(knots::ReverseDiff.TrackedArray, x::TrackedMatrix)
    return NormalizingFlows._rqs_bin(value(knots), value(x))
end

NormalizingFlows._rqs_nonneg(b::ReverseDiff.TrackedArray) = value(b) .>= 0

end
