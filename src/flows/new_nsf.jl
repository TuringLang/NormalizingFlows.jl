using MonotonicSplines

struct NSC{T,A<:Flux.Chain} <: Bijectors.Bijector
    dim::Int                        # dimension of input
    K::Int                          # number of knots
    n_dims_transferred::Int         # number of dimensions that are transformed
    B::T                            # bound of the knots
    nn::A                           # networks that parmaterize the knots and derivatives
    mask::Bijectors.PartitionMask
end

function NSC(
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
    return NSC{T2, typeof(nn)}(dim, K, num_of_transformed_dims, B, nn, mask)
end

@functor NSC (nn,)

function get_nsl_params(nsl::NSC, x::AbstractVecOrMat)
    nnoutput = nsl.nn(x)
    px, py, dydx = MonotonicSplines.rqs_params_from_nn(nnoutput, nsl.n_dims_transferred, nsl.B)
    return px, py, dydx
end

function Bijectors.transform(nsl::NSC, x::AbstractVecOrMat)
    x1, x2, x3 = Bijectors.partition(nsl.mask, x)
    # instantiate rqs knots and derivatives
    px, py, dydx = get_nsl_params(nsl, x2)
    if x1 isa AbstractVector
        x1 = reshape(x1, 1, length(x1))  # ensure x1 is a matrix
    end
    y1, _ = MonotonicSplines.rqs_forward(x1, px, py, dydx)
    return Bijectors.combine(nsl.mask, y1, x2, x3)
end

function Bijectors.with_logabsdet_jacobian(nsl::NSC, x::AbstractVecOrMat)
    x1, x2, x3 = Bijectors.partition(nsl.mask, x)
    # instantiate rqs knots and derivatives
    px, py, dydx = get_nsl_params(nsl, x2)
    y1, logjac = MonotonicSplines.rqs_forward(x1, px, py, dydx)
    return Bijectors.combine(nsl.mask, y1, x2, x3), vec(logjac)
end

function Bijectors.transform(insl::Inverse{<:NSC}, y::AbstractVecOrMat)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    px, py, dydx = get_nsl_params(nsl, y2)
    x1, _ = MonotonicSplines.rqs_inverse(y1, px, py, dydx)
    return Bijectors.combine(nsl.mask, x1, y2, y3)
end

function Bijectors.with_logabsdet_jacobian(insl::Inverse{<:NSC}, y::AbstractVecOrMat)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    px, py, dydx = get_nsl_params(nsl, y2)
    x1, logjac = MonotonicSplines.rqs_inverse(y1, px, py, dydx)
    return Bijectors.combine(nsl.mask, x1, y2, y3), logjac isa Real ? logjac : vec(logjac)
end

function (nsl::NSC)(x::AbstractVecOrMat)
    return Bijectors.transform(nsl, x)
end


function new_NSF_layer(
    dims::T1,                      # dimension of problem
    hdims::AbstractVector{T1},     # dimension of hidden units for nn 
    K::T1,                           # number of knots
    B::T2;                           # bound of the knots
    paramtype::Type{T2} = Float64,   # type of the parameters
) where {T1<:Int,T2<:AbstractFloat}

    mask_idx1 = 1:2:dims
    mask_idx2 = 2:2:dims

    # by default use the odd-even masking strategy
    nsf1 = NSC(dims, hdims, K, B, mask_idx1, paramtype)
    nsf2 = NSC(dims, hdims, K, B, mask_idx2, paramtype)
    return reduce(∘, (nsf1, nsf2))
end

function new_nsf(
    q0::Distribution{Multivariate,Continuous},  
    hdims::AbstractVector{Int},     # dimension of hidden units for s and t
    K::Int,
    B::T,
    nlayers::Int;                   # number of RealNVP_layer 
    paramtype::Type{T} = Float64,   # type of the parameters
) where {T<:AbstractFloat}

    dims = length(q0)  # dimension of the reference distribution == dim of the problem
    Ls = [new_NSF_layer(dims, hdims, K, B; paramtype=paramtype) for _ in 1:nlayers] 
    create_flow(Ls, q0)         
end

new_nsf(q0; paramtype::Type{T} = Float64) where {T<:AbstractFloat} = new_nsf(
    q0, [32, 32], 10, 30*one(T), 10; paramtype=paramtype
)
