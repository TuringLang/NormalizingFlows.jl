module NormalizingFlowsEnzymeExt

if isdefined(Base, :get_extension)
    using Enzyme
    using NormalizingFlows
    using NormalizingFlows: ADTypes, DiffResults
else
    using ..Enzyme
    using ..NormalizingFlows
    using ..NormalizingFlows: ADTypes, DiffResults
end

function NormalizingFlows.value_and_gradient!(
    ad::ADTypes.AutoEnzyme, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    ∇θ = DiffResults.gradient(out)
    fill!(∇θ, zero(T))
    _, y = Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ))
    DiffResults.value!(out, y)
    return out
end

end
