using Flux
using Functors
using Bijectors
using Bijectors: partition, combine, PartitionMask

include("../util.jl")

"""
Affinecoupling layer for RealNVP "(http://proceedings.mlr.press/v118/fjelde20a/fjelde20a.pdf)"
"""
struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# let params track field s and t
@functor AffineCoupling (s, t)

function AffineCoupling(
    dim::Int,  # dimension of input
    hdims::Int, # dimension of hidden units for s and t
    mask_idx::AbstractVector, # index of dimensione that one wants to apply transformations on
)
    cdims = length(mask_idx) # dimension of parts used to construct coupling law
    s = MLP_3layer(cdims, hdims, cdims)
    t = MLP_3layer(cdims, hdims, cdims)
    mask = PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end

## scaling parameterize using exp
# function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
#     # partition vector using 'af.mask::PartitionMask`
#     x₁, x₂, x₃ = Bijectors.partition(af.Mask, x)
#     y₁ = x₁ .* exp.(af.s(x₂)) .+ af.t(x₂)
#     return Bijectors.combine(af.mask, y₁, x₂, x₃)
# end

function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃ = partition(af.mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return combine(af.mask, y₁, x₂, x₃)
end

function (af::AffineCoupling)(x::AbstractArray)
    return transform(af, x)
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = exp.(af.s(x_2)) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2))
    return combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractVector
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    logjac = -sum(log ∘ abs, af.s(y_2))
    return combine(af.mask, x_1, y_2, y_3), logjac
end

function Bijectors.logabsdetjac(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = partition(af.mask, x)
    logjac = sum(log ∘ abs, af.s(x_2))
    return logjac
end

################### 
# a equivalent definition of AffineCoupling using Bijectors.Coupling 
# (see https://github.com/TuringLang/Bijectors.jl/blob/74d52d4eda72a6149b1a89b72524545525419b3f/src/bijectors/coupling.jl#L188C1-L188C1)
###################

# struct AffineCoupling <: Bijectors.Bijector
#     dim::Int
#     mask::Bijectors.PartitionMask
#     s::Flux.Chain
#     t::Flux.Chain
# end

# # let params track field s and t
# @functor AffineCoupling (s, t)

# function AffineCoupling(dim, mask, s, t)
#     return Bijectors.Coupling(θ -> Bijectors.Shift(t(θ)) ∘ Bijectors.Scale(s(θ)), mask)
# end

# function AffineCoupling(
#     dim::Int,  # dimension of input
#     hdims::Int, # dimension of hidden units for s and t
#     mask_idx::AbstractVector, # index of dimensione that one wants to apply transformations on
# )
#     cdims = length(mask_idx) # dimension of parts used to construct coupling law
#     s = MLP_3layer(cdims, hdims, cdims)
#     t = MLP_3layer(cdims, hdims, cdims)
#     mask = PartitionMask(dim, mask_idx)
#     return AffineCoupling(dim, mask, s, t)
# end
