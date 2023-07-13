module NormalizingFlowsZygoteExt

if isdefined(Base, :get_extension)
    using NormalizingFlows
    using NormalizingFlows: AutoZygote, MutableDiffResult, value!, gradient!
    using Zygote
else
    using ..NormalizingFlows
    using ..Zygote
end

@info "loading zygote ext"
function NormalizingFlows.value_and_gradient!(
    ad::AutoZygote, f, θ::AbstractVector{T}, out::MutableDiffResult
) where {T<:Real}
    y, back = Zygote.pullback(f, θ)
    ∇θ = back(one(T))
    value!(out, y)
    gradient!(out, first(∇θ))
    return out
end

end