"""
Neural Rational quadratic Spline layer 

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
    mask_idx::AbstractVector{<:Int}, # index of dimensione that one wants to apply transformations on
    paramtype::Type{T3},             # type of the parameters, e.g., Float64 or Float32
) where {T1<:Int,T2<:Real,T3<:AbstractFloat}
    num_of_transformed_dims = length(mask_idx)
    input_dims = dim - num_of_transformed_dims
    
    # output dim of the NN
    output_dims = (3K - 1)*num_of_transformed_dims
    # one big mlp that outputs all the knots and derivatives for all the transformed dimensions
    nn = fnn(input_dims, hdims, output_dims; output_activation=nothing, paramtype=paramtype)

    mask = Bijectors.PartitionMask(dim, mask_idx)
    return NeuralSplineCoupling(dim, K, num_of_transformed_dims, B, nn, mask)
end

@functor NeuralSplineCoupling (nn,)

"""
Build a rational quadratic spline (RQS) from the nn output
Bijectors.jl has implemented the inverse and logabsdetjac for rational quadratic spline

we just need to map the nn output to the knots and derivatives of the RQS
"""
function instantiate_rqs(nsl::NeuralSplineCoupling, x::AbstractVector)
    K, B = nsl.K, nsl.B
    nnoutput = reshape(nsl.nn(x), nsl.n_dims_transferred, :)
    ws = @view nnoutput[:, 1:K]
    hs = @view nnoutput[:, (K + 1):(2K)]
    ds = @view nnoutput[:, (2K + 1):(3K - 1)]
    return Bijectors.RationalQuadraticSpline(ws, hs, ds, B)
end

function Bijectors.transform(nsl::NeuralSplineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    # instantiate rqs knots and derivatives
    rqs = instantiate_rqs(nsl, x_2)
    y_1 = Bijectors.transform(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3)
end

function Bijectors.transform(insl::Inverse{<:NeuralSplineCoupling}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, y3 = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    x1 = Bijectors.transform(Inverse(rqs), y1)
    return Bijectors.combine(nsl.mask, x1, y2, y3)
end

function (nsl::NeuralSplineCoupling)(x::AbstractVector)
    return Bijectors.transform(nsl, x)
end

# define logabsdetjac
function Bijectors.logabsdetjac(nsl::NeuralSplineCoupling, x::AbstractVector)
    x_1, x_2, _ = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    logjac = logabsdetjac(rqs, x_1)
    return logjac
end

function Bijectors.logabsdetjac(insl::Inverse{<:NeuralSplineCoupling}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, _ = partition(nsl.mask, y)
    rqs = instantiate_rqs(nsl, y2)
    logjac = logabsdetjac(Inverse(rqs), y1)
    return logjac
end

function Bijectors.with_logabsdet_jacobian(nsl::NeuralSplineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    y_1, logjac = with_logabsdet_jacobian(rqs, x_1)
    return Bijectors.combine(nsl.mask, y_1, x_2, x_3), logjac
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
