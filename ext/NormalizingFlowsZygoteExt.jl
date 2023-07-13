module NormalizingFlowsZygoteExt

if isdefined(Base, :get_extension)
    using Zygote
    using NormalizingFlows: value_and_gradient!
    using ADTypes
    using DiffResults
else
    using ..Zygote
    using ..NormalizingFlows: value_and_gradient!
    using ..ADTypes
    using ..DiffResults
end

@info "loading zygote ext"
function value_and_gradient!(
    ad::ADTypes.AutoZygote, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(T))
    DiffResults.value!(out, y)
    DiffResults.gradient!(out, first(∇θ))
    return out
end

end