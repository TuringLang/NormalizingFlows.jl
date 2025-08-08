"""
Affine coupling layer used in RealNVP.

Implements two subnetworks `s` (scale, exponentiated) and `t` (shift) applied to
one partition of the input, conditioned on the complementary partition. The
scale network uses `tanh` on its output before exponentiation to improve
stability during training.

See also: Dinh et al., 2016 (RealNVP).
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
    dim::Int,                       # dimension of the problem
    hdims::AbstractVector{Int},     # dimension of hidden units for s and t
    mask_idx::AbstractVector{Int},       # index of dimensione that one wants to apply transformations on
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

################### 
# an equivalent definition of AffineCoupling using Bijectors.Coupling 
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
#     s = mlp3(cdims, hdims, cdims)
#     t = mlp3(cdims, hdims, cdims)
#     mask = PartitionMask(dim, mask_idx)
#     return AffineCoupling(dim, mask, s, t)
# end

"""
    RealNVP_layer(dims, hdims; paramtype = Float64)

Construct a single RealNVP layer using two affine coupling bijections with
odd–even masks.

Arguments
- `dims::Int`: dimensionality of the target distribution
- `hdims::AbstractVector{Int}`: hidden sizes for the `s` and `t` MLPs

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type

Returns
- A `Bijectors.Bijector` representing the RealNVP layer
"""
function RealNVP_layer(
    dims::Int,                      # dimension of problem
    hdims::AbstractVector{Int};     # dimension of hidden units for s and t
    paramtype::Type{T} = Float64,   # type of the parameters
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

Construct a RealNVP flow by stacking `nlayers` RealNVP_layer blocks with
odd–even masking. The no-argument variant uses 10 layers with `[32, 32]`
hidden sizes per coupling network.

Arguments
- `q0::Distribution{Multivariate,Continuous}`: base distribution (e.g. `MvNormal(zeros(d), I)`)
- `hdims::AbstractVector{Int}`: hidden sizes for the `s` and `t` MLPs
- `nlayers::Int`: number of RealNVP layers

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type

Returns
- `Bijectors.MultivariateTransformed` representing the RealNVP flow
"""
function realnvp(
    q0::Distribution{Multivariate,Continuous},  
    hdims::AbstractVector{Int},     # dimension of hidden units for s and t
    nlayers::Int;                   # number of RealNVP_layer 
    paramtype::Type{T} = Float64,   # type of the parameters
) where {T<:AbstractFloat}

    dims = length(q0)  # dimension of the reference distribution == dim of the problem
    Ls = [RealNVP_layer(dims, hdims; paramtype=paramtype) for _ in 1:nlayers] 
    create_flow(Ls, q0)         
end

"""
    realnvp(q0; paramtype = Float64)

Default constructor of RealNVP with 10 layers, 
each coupling function has 2 hidden layers with 32 units. 
Following the general architecture as in [^ASD2020] (see Apdx. E).


[^ASD2020]: Agrawal, A., & Sheldon, D., & Domke, J. (2020). 
Advances in Black-Box VI: Normalizing Flows, Importance Weighting, and Optimization. 
In *NeurIPS*.
"""
realnvp(q0; paramtype::Type{T} = Float64) where {T<:AbstractFloat} = realnvp(
    q0, [32, 32], 10; paramtype=paramtype
)
