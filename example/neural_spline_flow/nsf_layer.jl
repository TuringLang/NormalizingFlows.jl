using Flux
using Functors
using Bijectors
using Bijectors: partition, PartitionMask

include("../util.jl")
"""
Neural Rational quadratic Spline layer 

# References
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). 
"""
# struct NeuralSplineLayer{T1,T2,A<:AbstractVecOrMat{T1}} <: Bijectors.Bijector
#     dim::Int
#     mask::Bijectors.PartitionMask
#     w::A # width 
#     h::A # height 
#     d::A # derivative of the knots
#     B::T2 # bound of the knots
# end

# function NeuralSplineLayer(
#     dim::Int,  # dimension of input
#     hdims::Int, # dimension of hidden units for s and t
#     K::Int, # number of knots
#     B::T2, # bound of the knots
#     mask_idx::AbstractVector{<:Int}, # index of dimensione that one wants to apply transformations on
# ) where {T2<:Real}
#     num_of_transformed_dims = length(mask_idx)
#     input_dims = dim - num_of_transformed_dims
#     w = fill(MLP_3layer(input_dims, hdims, K), num_of_transformed_dims)
#     h = fill(MLP_3layer(input_dims, hdims, K), num_of_transformed_dims)
#     d = fill(MLP_3layer(input_dims, hdims, K - 1), num_of_transformed_dims)
#     mask = Bijectors.PartitionMask(dim, mask_idx)
#     return NeuralSplineLayer(dim, mask, w, h, d, B)
# end

# @functor NeuralSplineLayer (w, h, d)

# # define forward and inverse transformation
# function instantiate_rqs(nsl::NeuralSplineLayer, x::AbstractVector)
#     # instantiate rqs knots and derivatives
#     ws = permutedims(reduce(hcat, [w(x) for w in nsl.w]))
#     hs = permutedims(reduce(hcat, [h(x) for h in nsl.h]))
#     ds = permutedims(reduce(hcat, [d(x) for d in nsl.d]))
#     return Bijectors.RationalQuadraticSpline(ws, hs, ds, nsl.B)
# end

## Question: which one is better, the struct below or the struct above?
struct NeuralSplineLayer{T,A<:Flux.Chain} <: Bijectors.Bijector
    dim::Int
    K::Int
    nn::AbstractVector{A} # networks that parmaterize the knots and derivatives
    B::T # bound of the knots
    mask::Bijectors.PartitionMask
end

function NeuralSplineLayer(
    dim::T1,  # dimension of input
    hdims::T1, # dimension of hidden units for s and t
    K::T1, # number of knots
    B::T2, # bound of the knots
    mask_idx::AbstractVector{<:Int}, # index of dimensione that one wants to apply transformations on
) where {T1<:Int,T2<:Real}
    num_of_transformed_dims = length(mask_idx)
    input_dims = dim - num_of_transformed_dims
    nn = fill(MLP_3layer(input_dims, hdims, 3K - 1), num_of_transformed_dims)
    mask = Bijectors.PartitionMask(dim, mask_idx)
    return NeuralSplineLayer(dim, K, nn, B, mask)
end

@functor NeuralSplineLayer (nn,)

# define forward and inverse transformation
function instantiate_rqs(nsl::NeuralSplineLayer, x::AbstractVector)
    # instantiate rqs knots and derivatives
    T = permutedims(reduce(hcat, map(nn -> nn(x), nsl.nn)))
    K, B = nsl.K, nsl.B
    ws = T[:, 1:K]
    hs = T[:, (K + 1):(2K)]
    ds = T[:, (2K + 1):(3K - 1)]
    return Bijectors.RationalQuadraticSpline(ws, hs, ds, B)
end

function Bijectors.transform(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    # instantiate rqs knots and derivatives
    rqs = instantiate_rqs(nsl, x_2)
    y_1 = Bijectors.transform(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3)
end

function Bijectors.transform(insl::Inverse{<:NeuralSplineLayer}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    x1 = Bijectors.transform(Inverse(rqs), y1)
    return Bijectors.combine(nsl.mask, x1, y2, y3)
end

function (nsl::NeuralSplineLayer)(x::AbstractVector)
    return Bijectors.transform(nsl, x)
end

# define logabsdetjac
function Bijectors.logabsdetjac(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    logjac = logabsdetjac(rqs, x_1)
    return logjac
end

function Bijectors.logabsdetjac(insl::Inverse{<:NeuralSplineLayer}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    logjac = logabsdetjac(Inverse(rqs), y1)
    return logjac
end

function Bijectors.with_logabsdet_jacobian(nsl::NeuralSplineLayer, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    y_1, logjac = with_logabsdet_jacobian(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3), logjac
end
