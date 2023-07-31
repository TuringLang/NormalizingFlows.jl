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
    mask = Bijectors.PartitionMask(dim, mask_idx)
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
    x₁, x₂, x₃ = Bijectors.partition(af.Mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return Bijectors.combine(af.Mask, y₁, x₂, x₃)
end

function (af::AffineCoupling)(x::AbstractArray)
    return Bijectors.transform(af, x)
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = exp.(af.s(x_2)) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2))  # logabsdetjac of exp is simply the argument
    return combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractVector
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.Mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) .* exp.(-af.s(y_2))
    logjac = -sum(af.s(y_2))
    return combine(af.Mask, x_1, y_2, y_3), logjac
end

function Bijectors.logabsdetjac(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = partition(af.Mask, x)
    logjac = sum(af.s(x_2))
    return logjac
end

# # test invertibility
# Ls =  [
#     AffineCoupling(4, 8, 1:2),
#     AffineCoupling(4, 8, 3:4),
#     AffineCoupling(4, 8, 1:2),
#     AffineCoupling(4, 8, 3:4),
# ]
# T = Bijectors.∘(Ls...)
# B = inverse(T)

# x0 = randn(4)
# x00 = B(T(x0))  
# norm(x0 .- x00) 

# # test autograd
# Flux.params(T)
# function test(x)
#     sum(abs2,T(x))
# end

# using Zygote
# Zygote.gradient(test, x0)

# X = randn(2)
# mask = Bijectors.PartitionMask(2, 1:1)
# x1, x2, x3 = Bijectors.partition(mask, X)