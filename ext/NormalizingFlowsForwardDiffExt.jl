module NormalizingFlowsForwardDiffExt

if isdefined(Base, :get_extension)
    using ForwardDiff
    using NormalizingFlows: value_and_gradient!
    using ADTypes
    using DiffResults
else
    using ..ForwardDiff
    using ..NormalizingFlows: value_and_gradient!
    using ..ADTypes
    using ..DiffResults
end

# ForwardDiff
# extract chunk size from AutoForwardDiff
getchunksize(::ADTypes.AutoForwardDiff{chunksize}) where {chunksize} = chunksize
function value_and_gradient!(
    ad::ADTypes.AutoForwardDiff, f, θ::AbstractVector{T}, out::DiffResults.MutableDiffResult
) where {T<:Real}
    chunk_size = getchunksize(ad)
    config = if isnothing(chunk_size)
        ForwardDiff.GradientConfig(f, θ)
    else
        ForwardDiff.GradientConfig(f, θ, ForwardDiff.Chunk(length(θ), chunk_size))
    end
    ForwardDiff.gradient!(out, f, θ, config)
    return out
end

end