using Statistics: mean, std
using Functors
using Bijectors

mutable struct InvertibleAffine <: Bijectors.Bijector
    shift
    scale
end

@functor InvertibleAffine

InvertibleAffine() = InvertibleAffine(nothing, nothing)

function Bijectors.transform(af::InvertibleAffine, xs::AbstractMatrix)
    if af.shift === nothing && af.scale === nothing
        af.shift = vec(mean(xs; dims=2))
        af.scale = 1 ./ vec(std(xs; dims=2))
    end
    return af.scale .* (xs .- af.shift)
end

function Bijectors.with_logabsdet_jacobian(af::InvertibleAffine, xs::AbstractMatrix)
    if af.shift === nothing && af.scale === nothing
        af.shift = vec(mean(xs; dims=2))
        af.scale = 1 ./ vec(std(xs; dims=2))
    end
    ys = af.scale .* (xs .- af.shift)
    logjacs = sum(log ∘ abs, af.scale) * ones(size(xs, 2))
    return ys, logjacs
end

function Bijectors.transform(iaf::Inverse{<:InvertibleAffine}, ys::AbstractMatrix)
    af = iaf.orig
    if af.shift === nothing && af.scale === nothing
        throw(ArgumentError("shift and scale must be specified"))
    end
    return ys ./ af.scale .+ af.shift
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:InvertibleAffine}, ys::AbstractMatrix
)
    af = iaf.orig
    if af.shift === nothing && af.scale === nothing
        throw(ArgumentError("shift and scale must be specified"))
    end
    xs = ys ./ af.scale .+ af.shift
    logjacs = sum(log ∘ abs, af.scale) * ones(size(ys, 2))
    return ys, -logjacs
end

mutable struct InvertibleAffineBwd <: Bijectors.Bijector
    shift
    scale
end

@functor InvertibleAffineBwd

InvertibleAffineBwd() = InvertibleAffineBwd(nothing, nothing)

function Bijectors.transform(af::InvertibleAffineBwd, xs::AbstractMatrix)
    if af.shift === nothing && af.scale === nothing
        throw(ArgumentError("shift and scale must be specified"))
    end
    return xs ./ af.scale .+ af.shift
end

function Bijectors.with_logabsdet_jacobian(af::InvertibleAffineBwd, xs::AbstractMatrix)
    if af.shift === nothing && af.scale === nothing
        throw(ArgumentError("shift and scale must be specified"))
    end
    logjacs = sum(log ∘ abs, af.scale) * ones(size(xs, 2))
    return ys, -logjacs
end

function Bijectors.transform(iaf::Inverse{<:InvertibleAffineBwd}, ys::AbstractMatrix)
    af = iaf.orig
    if af.shift === nothing && af.scale === nothing
        af.shift = vec(mean(ys; dims=2))
        af.scale = 1 ./ vec(std(ys; dims=2))
    end
    return (ys .- af.shift) .* af.scale
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:InvertibleAffine}, ys::AbstractMatrix
)
    af = iaf.orig
    if af.shift === nothing && af.scale === nothing
        af.shift = vec(mean(ys; dims=2))
        af.scale = 1 ./ vec(std(ys; dims=2))
    end
    xs = (ys .- af.shift) .* af.scale
    logjacs = sum(log ∘ abs, af.scale) * ones(size(ys, 2))
    return ys, logjacs
end
