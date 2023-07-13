module NormalizingFlowsEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme
    using NormalizingFlows
    using NormalizingFlows: AutoEnzyme, MutableDiffResult, value!, gradient
else
    using ..Enzyme
    using ..NormalizingFlows
    using ..NormalizingFlows: AutoEnzyme, MutableDiffResult, value!, gradient
end

# Enzyme doesn't support f::Bijectors (see https://github.com/EnzymeAD/Enzyme.jl/issues/916)
function NormalizingFlows.value_and_gradient!(
    ad::AutoEnzyme, f, θ::AbstractVector{T}, out::MutableDiffResult
) where {T<:Real}
    y = f(θ)
    value!(out, y)
    ∇θ = gradient(out)
    fill!(∇θ, zero(T))
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ))
    return out
end

end