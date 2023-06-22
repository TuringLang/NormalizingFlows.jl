using Flux
using Functors
using Bijectors
using Bijectors: partition, PartitionMask
"""
Neural Rational quadratic Spline layer "(https://proceedings.neurips.cc/paper_files/paper/2019/file/7ac71d433f282034e088473244df8c02-Paper.pdf)"
"""
struct NeuralSplineLayer{T} <: Bijectors.Bijector
    D::Int
    mask::Bijectors.PartitionMask
    w::T # width (xs)
    h::T # height (ys)
    d::T # derivative of the knots
    B::Real # bound of the knots
end

function MLP_3layer(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim),
    )
end

function NeuralSplineLayer(
    D::Int,  # dimension of input
    hdims::Int, # dimension of hidden units for s and t
    K::Int, # number of knots
    mask_idx::AbstractVector, # index of dimensione that one wants to apply transformations on
    B::Real, # bound of the knots
)
    num_of_transformed_dims = length(mask_idx)
    input_dims = D - num_of_transformed_dims
    w = [MLP_3layer(input_dims, hdims, K) for i in 1:num_of_transformed_dims]
    h = [MLP_3layer(input_dims, hdims, K) for i in 1:num_of_transformed_dims]
    d = [MLP_3layer(input_dims, hdims, K - 1) for i in 1:num_of_transformed_dims]
    mask = Bijectors.PartitionMask(D, mask_idx)
    return NeuralSplineLayer(D, mask, w, h, d, B)
end

@functor NeuralSplineLayer (w, h, d)

# define forward and inverse transformation
function instantiate_rqs(nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector)
    # instantiate rqs knots and derivatives
    ws = reduce(hcat, [w(x) for w in nsl.w])
    hs = reduce(hcat, [h(x) for h in nsl.h])
    ds = reduce(hcat, [d(x) for d in nsl.d])
    # TODO: need to ask whether there is a better way
    return Bijectors.RationalQuadraticSpline(ws', hs', ds', nsl.B)
end

function Bijectors.transform(
    nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector
)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    # TODO: need to ask whether there is a better way
    # instantiate rqs knots and derivatives
    rqs = instantiate_rqs(nsl, x_2)
    y_1 = transform(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3)
end

function Bijectors.transform(
    insl::Inverse{<:NeuralSplineLayer{<:Vector{<:Flux.Chain}}}, y::AbstractVector
)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    # todo: improve
    rqs = instantiate_rqs(nsl, y2)
    x1 = transform(Inverse(rqs), y1)
    return combine(nsl.mask, x1, y2, y3)
end

function (nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}})(x::AbstractVector)
    return Bijectors.transform(nsl, x)
end

# define logabsdetjac
function Bijectors.logabsdetjac(
    nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector
)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    Rqs = instantiate_rqs(nsl, x_2)
    logjac = logabsdetjac(Rqs, x_1)
    return logjac
end

function Bijectors.logabsdetjac(
    insl::Inverse{<:NeuralSplineLayer{<:Vector{<:Flux.Chain}}}, y::AbstractVector
)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    logjac = logabsdetjac(Inverse(rqs), y1)
    return logjac
end

function Bijectors.with_logabsdet_jacobian(
    nsl::NeuralSplineLayer{<:Vector{<:Flux.Chain}}, x::AbstractVector
)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    y_1, logjac = with_logabsdet_jacobian(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3), logjac
end
