# a new implementation of Neural Spline Flow based on MonotonicSplines.jl
# the construction of the RQS seems to be more efficient than the one in Bijectors.jl
# and supports batched operations.

"""
Neural Rational Quadratic Spline Coupling layer 
# References
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). 
"""
struct NeuralSplineCoupling{T,A<:Flux.Chain} <: Bijectors.Bijector
    dim::Int                        # dimension of input
    K::Int                          # number of knots
    n_dims_transferred::Int         # number of dimensions that are transformed
    B::T                            # bound of the knots
    nn::A                           # networks that parmaterize the knots and derivatives
    mask::Bijectors.PartitionMask
end

function NeuralSplineCoupling(
    dim::T1,                         # dimension of input
    hdims::AbstractVector{T1},       # dimension of hidden units for s and t
    K::T1,                           # number of knots
    B::T2,                           # bound of the knots
    mask_idx::AbstractVector{T1}, # index of dimensione that one wants to apply transformations on
    paramtype::Type{T2},             # type of the parameters, e.g., Float64 or Float32
) where {T1<:Int,T2<:AbstractFloat}
    num_of_transformed_dims = length(mask_idx)
    input_dims = dim - num_of_transformed_dims
    
    # output dim of the NN
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
        nnoutput, nsc.n_dims_transferred, nsc.B
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
    NSF_layer(dims, hdims; paramtype = Float64)
Default constructor of single layer of Neural Spline Flow (NSF) 
which is a composition of 2 neural spline coupling transformations with complementary masks.
The masking strategy is odd-even masking.
# Arguments
- `dims::Int`: dimension of the problem
- `hdims::AbstractVector{Int}`: dimension of hidden units for s and t
- `K::Int`: number of knots
- `B::AbstractFloat`: bound of the knots
# Keyword Arguments
- `paramtype::Type{T} = Float64`: type of the parameters, defaults to `Float64`
# Returns
- A `Bijectors.Bijector` representing the NSF layer.
"""
function NSF_layer(
    dims::T1,                      # dimension of problem
    hdims::AbstractVector{T1},     # dimension of hidden units for nn 
    K::T1,                           # number of knots
    B::T2;                           # bound of the knots
    paramtype::Type{T2} = Float64,   # type of the parameters
) where {T1<:Int,T2<:AbstractFloat}

    mask_idx1 = 1:2:dims
    mask_idx2 = 2:2:dims

    # by default use the odd-even masking strategy
    nsf1 = NeuralSplineCoupling(dims, hdims, K, B, mask_idx1, paramtype)
    nsf2 = NeuralSplineCoupling(dims, hdims, K, B, mask_idx2, paramtype)
    return reduce(∘, (nsf1, nsf2))
end

function nsf(
    q0::Distribution{Multivariate,Continuous},  
    hdims::AbstractVector{Int},     # dimension of hidden units for s and t
    K::Int,
    B::T,
    nlayers::Int;                   # number of RealNVP_layer 
    paramtype::Type{T} = Float64,   # type of the parameters
) where {T<:AbstractFloat}

    dims = length(q0)  # dimension of the reference distribution == dim of the problem
    Ls = [NSF_layer(dims, hdims, K, B; paramtype=paramtype) for _ in 1:nlayers] 
    create_flow(Ls, q0)         
end

nsf(q0; paramtype::Type{T} = Float64) where {T<:AbstractFloat} = nsf(
    q0, [32, 32], 10, 30*one(T), 10; paramtype=paramtype
)
