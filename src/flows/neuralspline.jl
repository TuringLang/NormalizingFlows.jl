"""
    NeuralSplineCoupling(dim, hdims, K, B, mask_idx, paramtype)
    NeuralSplineCoupling(dim, K, n_dims_transformed, B, nn, mask)

Neural Rational Quadratic Spline (RQS) coupling bijector [^DBMP2019].

A conditioner network takes the unchanged partition as input and outputs the
parameters of monotonic rational quadratic splines for the transformed
coordinates. Batched inputs (matrices with column vectors) are supported.

Arguments
- `dim::Int`: total input dimension.
- `hdims::AbstractVector{Int}`: hidden sizes for the conditioner MLP.
- `K::Int`: number of spline knots per transformed coordinate.
- `B::AbstractFloat`: boundary/box constraint for spline domain.
- `mask_idx::AbstractVector{Int}`: indices of the transformed coordinates.

Keyword Arguments
- `paramtype::Type{<:AbstractFloat}`: parameter element type.

Fields
- `nn::Flux.Chain`: conditioner that outputs all spline params for all transformed dim.
- `mask::Bijectors.PartitionMask`: partition specification.

Notes
- Output dimensionality of the conditioner is `(3K - 1) * n_transformed`.
- For computation performance, we rely on 
[`MonotonicSplines.jl`](https://github.com/bat/MonotonicSplines.jl) for the
building the rational quadratic spline functions.
- See `MonotonicSplines.rqs_forward` and `MonotonicSplines.rqs_inverse` for forward/inverse 
and log-determinant computations.

[^DBMP2019]: Durkan, C., Bekasov, A., Murray, I. and Papamarkou, T. (2019). Neural Spline Flows. *NeurIPS.*
"""
struct NeuralSplineCoupling{T,A<:Flux.Chain} <: Bijectors.Bijector
    dim::Int                        # dimension of input
    K::Int                          # number of knots
    n_dims_transformed::Int         # number of dimensions that are transformed
    B::T                            # bound of the knots
    nn::A                           # networks that parameterize the knots and derivatives
    mask::Bijectors.PartitionMask
end

function NeuralSplineCoupling(
    dim::T1,                         # dimension of input
    hdims::AbstractVector{T1},       # dimension of hidden units for s and t
    K::T1,                           # number of knots
    B::T2,                           # bound of the knots
    mask_idx::AbstractVector{T1},    # indices of the transformed dimensions
    paramtype::Type{T2},             # type of the parameters, e.g., Float64 or Float32
) where {T1<:Int,T2<:AbstractFloat}
    num_of_transformed_dims = length(mask_idx)
    input_dims = dim - num_of_transformed_dims
    
    output_dims = (3K - 1)*num_of_transformed_dims
    # one big mlp that outputs all the knots and derivatives for all the transformed dimensions
    nn = fnn(input_dims, hdims, output_dims; output_activation=nothing, paramtype=paramtype)

    mask = Bijectors.PartitionMask(dim, mask_idx)
    return NeuralSplineCoupling{T2, typeof(nn)}(dim, K, num_of_transformed_dims, B, nn, mask)
end

@functor NeuralSplineCoupling (nn,)

function get_nsc_params(nsc::NeuralSplineCoupling, x::AbstractVecOrMat)
    nnoutput = nsc.nn(x)
    px, py, dydx = MonotonicSplines.rqs_params_from_nn(
        nnoutput, nsc.n_dims_transformed, nsc.B
    )
    return px, py, dydx
end

# when input x is a vector instead of a matrix
# need this to transform it to a matrix with one row
# otherwise, rqs_forward and rqs_inverse will throw an error
_ensure_matrix(x) = x isa AbstractVector ? reshape(x, length(x), 1) : x

function Bijectors.transform(nsc::NeuralSplineCoupling, x::AbstractVector)
    x1, x2, x3 = Bijectors.partition(nsc.mask, x)
    # instantiate rqs knots and derivatives
    px, py, dydx = get_nsc_params(nsc, x2)
    x1 = _ensure_matrix(x1)
    y1, _ = MonotonicSplines.rqs_forward(x1, px, py, dydx)
    return Bijectors.combine(nsc.mask, vec(y1), x2, x3)
end
function Bijectors.transform(nsc::NeuralSplineCoupling, x::AbstractMatrix)
    x1, x2, x3 = Bijectors.partition(nsc.mask, x)
    # instantiate rqs knots and derivatives
    px, py, dydx = get_nsc_params(nsc, x2)
    y1, _ = MonotonicSplines.rqs_forward(x1, px, py, dydx)
    return Bijectors.combine(nsc.mask, y1, x2, x3)
end

function Bijectors.with_logabsdet_jacobian(nsc::NeuralSplineCoupling, x::AbstractVector)
    x1, x2, x3 = Bijectors.partition(nsc.mask, x)
    # instantiate rqs knots and derivatives
    px, py, dydx = get_nsc_params(nsc, x2)
    x1 = _ensure_matrix(x1)
    y1, logjac = MonotonicSplines.rqs_forward(x1, px, py, dydx)
    return Bijectors.combine(nsc.mask, vec(y1), x2, x3), logjac[1]
end
function Bijectors.with_logabsdet_jacobian(nsc::NeuralSplineCoupling, x::AbstractMatrix)
    x1, x2, x3 = Bijectors.partition(nsc.mask, x)
    # instantiate rqs knots and derivatives
    px, py, dydx = get_nsc_params(nsc, x2)
    y1, logjac = MonotonicSplines.rqs_forward(x1, px, py, dydx)
    return Bijectors.combine(nsc.mask, y1, x2, x3), vec(logjac)
end

function Bijectors.transform(insl::Inverse{<:NeuralSplineCoupling}, y::AbstractVector)
    nsc = insl.orig
    y1, y2, y3 = partition(nsc.mask, y)
    px, py, dydx = get_nsc_params(nsc, y2)
    y1 = _ensure_matrix(y1)
    x1, _ = MonotonicSplines.rqs_inverse(y1, px, py, dydx)
    return Bijectors.combine(nsc.mask, vec(x1), y2, y3)
end
function Bijectors.transform(insl::Inverse{<:NeuralSplineCoupling}, y::AbstractMatrix)
    nsc = insl.orig
    y1, y2, y3 = partition(nsc.mask, y)
    px, py, dydx = get_nsc_params(nsc, y2)
    x1, _ = MonotonicSplines.rqs_inverse(y1, px, py, dydx)
    return Bijectors.combine(nsc.mask, x1, y2, y3)
end

function Bijectors.with_logabsdet_jacobian(insl::Inverse{<:NeuralSplineCoupling}, y::AbstractVector)
    nsc = insl.orig
    y1, y2, y3 = partition(nsc.mask, y)
    px, py, dydx = get_nsc_params(nsc, y2)
    y1 = _ensure_matrix(y1)
    x1, logjac = MonotonicSplines.rqs_inverse(y1, px, py, dydx)
    return Bijectors.combine(nsc.mask, vec(x1), y2, y3), logjac[1]
end
function Bijectors.with_logabsdet_jacobian(insl::Inverse{<:NeuralSplineCoupling}, y::AbstractMatrix)
    nsc = insl.orig
    y1, y2, y3 = partition(nsc.mask, y)
    px, py, dydx = get_nsc_params(nsc, y2)
    x1, logjac = MonotonicSplines.rqs_inverse(y1, px, py, dydx)
    return Bijectors.combine(nsc.mask, x1, y2, y3), vec(logjac)
end

function (nsc::NeuralSplineCoupling)(x::AbstractVecOrMat)
    return Bijectors.transform(nsc, x)
end


"""
    NSF_layer(dim, hdims, K, B; paramtype = Float64)

Build a single Neural Spline Flow (NSF) layer by composing two
`NeuralSplineCoupling` bijectors with complementary odd–even masks.

Arguments
- `dim::Int`: dimensionality of the problem.
- `hdims::AbstractVector{Int}`: hidden sizes of the conditioner network.
- `K::Int`: number of spline knots.
- `B::AbstractFloat`: spline boundary.

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type.

Returns
- A `Bijectors.Bijector` representing the NSF layer.

Example
- `layer = NSF_layer(4, [64,64], 10, 3.0)`
- `y = layer(randn(4, 32))`
"""
function NSF_layer(
    dim::T1,                      # dimension of problem
    hdims::AbstractVector{T1},     # dimension of hidden units for nn 
    K::T1,                           # number of knots
    B::T2;                           # bound of the knots
    paramtype::Type{T2} = Float64,   # type of the parameters
) where {T1<:Int,T2<:AbstractFloat}

    mask_idx1 = 1:2:dim
    mask_idx2 = 2:2:dim

    # by default use the odd-even masking strategy
    nsf1 = NeuralSplineCoupling(dim, hdims, K, B, mask_idx1, paramtype)
    nsf2 = NeuralSplineCoupling(dim, hdims, K, B, mask_idx2, paramtype)
    return reduce(∘, (nsf1, nsf2))
end

"""
    nsf(q0, hdims, K, B, nlayers; paramtype = Float64)
    nsf(q0; paramtype = Float64)

Construct an NSF by stacking `nlayers` `NSF_layer` blocks. The one-argument
variant defaults to 10 layers with `[32, 32]` hidden sizes, 10 knots, and
boundary `30` (scaled by `one(T)`).

Arguments
- `q0::Distribution{Multivariate,Continuous}`: base distribution.
- `hdims::AbstractVector{Int}`: hidden sizes of the conditioner network.
- `K::Int`: spline knots per coordinate.
- `B::AbstractFloat`: spline boundary.
- `nlayers::Int`: number of NSF layers.

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type.

Returns
- `Bijectors.TransformedDistribution` representing the NSF flow.

!!! note 
    Under the hood, `nsf` relies on the rational quadratic spline function implememented in 
    `MonotonicSplines.jl` for performance reasons.  `MonotonicSplines.jl` uses 
    `KernelAbstractions.jl` to support batched operations. 
    Because of this, so far `nsf` only supports `Zygote` as the AD type.
  

Example
- `q0 = MvNormal(zeros(3), I); flow = nsf(q0, [64,64], 8, 3.0, 6)`
- `x = rand(flow, 128); lp = logpdf(flow, x)`
"""
function nsf(
    q0::Distribution{Multivariate,Continuous},  
    hdims::AbstractVector{Int},     # dimension of hidden units for s and t
    K::Int,
    B::T,
    nlayers::Int;                   # number of RealNVP_layer 
    paramtype::Type{T} = Float64,   # type of the parameters
) where {T<:AbstractFloat}

    dim = length(q0)  # dimension of the reference distribution == dim of the problem
    Ls = [NSF_layer(dim, hdims, K, B; paramtype=paramtype) for _ in 1:nlayers] 
    create_flow(Ls, q0)         
end

nsf(q0; paramtype::Type{T} = Float64) where {T<:AbstractFloat} = nsf(
    q0, [32, 32], 10, 30*one(T), 10; paramtype=paramtype
)
