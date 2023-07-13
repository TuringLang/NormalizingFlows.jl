module NormalizingFlowsZygoteExt

if isdefined(Base, :get_extension)
    using ADTypes
    using DiffResults
    using NormalizingFlows
    using Zygote
else
    using ..ADTypes
    using ..DiffResults
    using ..NormalizingFlows
    using ..Zygote
end

@info "loading zygote ext"
function NormalizingFlows.value_and_gradient!(
    ad::ADTypes.AutoZygote, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(T))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, first(∇θ))
    return out
end

end