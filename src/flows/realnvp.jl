"""
    AffineCoupling(dim, hdims, mask_idx, paramtype)
    AffineCoupling(dim, mask, s, t)

Affine coupling bijector used in RealNVP [^LJS2017].

Two subnetworks `s` (log-scale, exponentiated in the forward pass) and `t` (shift)
act on one partition of the input, conditioned on the complementary partition
(as defined by `mask`). For numerical stability, the output of `s` passes
through `tanh` before exponentiation.

Arguments
- `dim::Int`: total dimensionality of the input.
- `hdims::AbstractVector{Int}`: hidden sizes for the conditioner MLPs `s` and `t`.
- `mask_idx::AbstractVector{Int}`: indices of the dimensions to transform.
  The complement is used as the conditioner input.

Keyword Arguments
- `paramtype::Type{<:AbstractFloat}`: parameter element type (e.g. `Float32`).

Fields
- `mask::Bijectors.PartitionMask`: partition specification.
- `s::Flux.Chain`: conditioner producing log-scales for the transformed block.
- `t::Flux.Chain`: conditioner producing shifts for the transformed block.

Notes
- Forward: with `(x₁,x₂,x₃) = partition(mask, x)`, compute `y₁ = x₁ .* exp.(s(x₂)) .+ t(x₂)`.
- Log-determinant: `sum(s(x₂))` (or columnwise for batched matrices).
- All methods support both vectors and column-major batches (matrices).

[^LJS2017]: Dinh, L., Sohl-Dickstein, J. and Bengio, S. (2017).  Density estimation using Real NVP. ICLR.
"""
struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

@functor AffineCoupling (s, t)

function AffineCoupling( 
    dim::Int,                       
    hdims::AbstractVector{Int},
    mask_idx::AbstractVector{Int},
    paramtype::Type{T}
) where {T<:AbstractFloat}
    cdims = length(mask_idx)  # dimension of parts used to construct coupling law
    # for the scaling network s, add tanh to the output to ensure stability during training
    s = fnn(dim-cdims, hdims, cdims; output_activation=Flux.tanh, paramtype=paramtype)  
    # no transfomration for the output of the translation network t
    t = fnn(dim-cdims, hdims, cdims; output_activation=nothing, paramtype=paramtype)
    mask = PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end

function Bijectors.transform(af::AffineCoupling, x::AbstractVecOrMat)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃ = partition(af.mask, x)
    s_x₂ = af.s(x₂)
    y₁ = x₁ .* exp.(s_x₂) .+ af.t(x₂)
    return combine(af.mask, y₁, x₂, x₃)
end

function (af::AffineCoupling)(x::AbstractVecOrMat)
    return transform(af, x)
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    s_x2 = af.s(x_2)
    y_1 = exp.(s_x2) .* x_1 .+ af.t(x_2)
    logjac = sum(s_x2) # this is a scalar
    return combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractMatrix)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    s_x2 = af.s(x_2)
    y_1 = exp.(s_x2) .* x_1 .+ af.t(x_2)
    logjac = sum(s_x2; dims=1) # 1 × size(x, 2)
    return combine(af.mask, y_1, x_2, x_3), vec(logjac)
end


function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractVector
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    s_y2 = af.s(y_2)
    x_1 = (y_1 .- af.t(y_2)) .* exp.(-s_y2)
    logjac = -sum(s_y2)
    return combine(af.mask, x_1, y_2, y_3), logjac
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractMatrix
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    s_y2 = af.s(y_2)
    x_1 = (y_1 .- af.t(y_2)) .* exp.(-s_y2)
    logjac = -sum(s_y2; dims=1)
    return combine(af.mask, x_1, y_2, y_3), vec(logjac)
end

"""
    RealNVP_layer(dims, hdims; paramtype = Float64)

Construct a single RealNVP layer by composing two `AffineCoupling` bijectors
with complementary odd–even masks.

Arguments
- `dims::Int`: dimensionality of the problem.
- `hdims::AbstractVector{Int}`: hidden sizes of the conditioner networks.

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type.

Returns
- A `Bijectors.Bijector` representing the RealNVP layer.

Example
- `layer = RealNVP_layer(4, [64, 64])`
- `y = layer(randn(4, 16))`  # batched forward
"""
function RealNVP_layer(
    dims::Int,
    hdims::AbstractVector{Int};
    paramtype::Type{T} = Float64,   
) where {T<:AbstractFloat}

    mask_idx1 = 1:2:dims
    mask_idx2 = 2:2:dims

    # by default use the odd-even masking strategy
    af1 = AffineCoupling(dims, hdims, mask_idx1, paramtype)
    af2 = AffineCoupling(dims, hdims, mask_idx2, paramtype)
    return reduce(∘, (af1, af2))
end

"""
    realnvp(q0, hdims, nlayers; paramtype = Float64)
    realnvp(q0; paramtype = Float64)

Construct a RealNVP flow by stacking `nlayers` `RealNVP_layer` blocks with
odd–even masking. The 1-argument variant defaults to 10 layers with
hidden sizes `[32, 32]` per conditioner.

Arguments
- `q0::Distribution{Multivariate,Continuous}`: base distribution (e.g. `MvNormal(zeros(d), I)`).
- `hdims::AbstractVector{Int}`: hidden sizes for the conditioner networks.
- `nlayers::Int`: number of stacked RealNVP layers.

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type (use `Float32` for GPU friendliness).

Returns
- `Bijectors.TransformedDistribution` representing the RealNVP flow.

Example
- `q0 = MvNormal(zeros(2), I); flow = realnvp(q0, [64,64], 8)`
- `x = rand(flow, 128); lp = logpdf(flow, x)`
"""
function realnvp(
    q0::Distribution{Multivariate,Continuous},  
    hdims::AbstractVector{Int},     
    nlayers::Int;
    paramtype::Type{T} = Float64,
) where {T<:AbstractFloat}

    dims = length(q0)
    Ls = [RealNVP_layer(dims, hdims; paramtype=paramtype) for _ in 1:nlayers] 
    create_flow(Ls, q0)         
end

"""
    realnvp(q0; paramtype = Float64)

Default constructor: 10 layers, each conditioner uses hidden sizes `[32, 32]`.
Follows a common RealNVP architecture similar to Appendix E of [^ASD2020].

[^ASD2020]: Agrawal, A., Sheldon, D., Domke, J. (2020). Advances in Black-Box VI: Normalizing Flows, Importance Weighting, and Optimization. NeurIPS.
"""
realnvp(q0; paramtype::Type{T} = Float64) where {T<:AbstractFloat} = realnvp(
    q0, [32, 32], 10; paramtype=paramtype
)
