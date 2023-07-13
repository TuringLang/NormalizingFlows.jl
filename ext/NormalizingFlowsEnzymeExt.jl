module NormalizingFlowsEnzymeExt

if isdefined(Base, :get_extension)
    using ADTypes
    using DiffResults
    using Enzyme
    using NormalizingFlows
else
    using ..ADTypes
    using ..DiffResults
    using ..Enzyme
    using ..NormalizingFlows
end

# Enzyme doesn't support f::Bijectors (see https://github.com/EnzymeAD/Enzyme.jl/issues/916)
function NormalizingFlows.value_and_gradient!(
    ad::ADTypes.AutoEnzyme, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y = f(θ)
    DiffResults.value!(out, y)
    ∇θ = DiffResults.gradient(out)
    fill!(∇θ, zero(T))
    Enzyme.autodiff(Enzyme.ReverseWithPrimal, f, Enzyme.Active, Enzyme.Duplicated(θ, ∇θ))
    return out
end

end