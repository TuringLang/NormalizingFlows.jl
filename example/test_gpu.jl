using Flux
using Bijectors
using Bijectors: partition, combine, PartitionMask

using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Mooncake
using NormalizingFlows

include("SyntheticTargets.jl")
include("utils.jl")

##################################
# define affine coupling layer using Bijectors.jl interface
#################################
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
    s = mlp3(cdims, hdims, cdims)
    t = mlp3(cdims, hdims, cdims)
    mask = PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end

function Bijectors.transform(af::AffineCoupling, x::AbstractVecOrMat)
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
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2)) # this is a scalar
    return combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractMatrix)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2); dims = 1) # 1 × size(x, 2)
    return combine(af.mask, y_1, x_2, x_3), vec(logjac)
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

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractMatrix
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    logjac = -sum(log ∘ abs, af.s(y_2); dims = 1)
    return combine(af.mask, x_1, y_2, y_3), vec(logjac)
end


##################################
# start demo
#################################
using CUDA
const NF = NormalizingFlows
rng_g = CUDA.default_rng() # use GPU RNG if available


CUDA.allowscalar(true)
n_samples = 100
q0 = MvNormal(CUDA.zeros(2), cu([1f0 0f0; 0f0 1f0]))
# gpu sample from the reference
xs = NF._device_specific_rand(rng_g, q0, n_samples)

d = 2
hdims = 32
Ls_g = [AffineCoupling(d, hdims, [1]) ∘ AffineCoupling(d, hdims, [2]) for i in 1:3]
flow_g = create_flow(Ls_g, q0)
flow_g = fmap(cu, flow_g) # move all flow parameters be on GPU

# gpu sample from the flow
ys = NF._device_specific_rand(rng_g, flow_g, n_samples)

# log density computation
logpdf(flow_g, ys) # errored

logpdf(q0, xs) # returns a CPU array


# elbo_batch(rng_g, flow, logp, n_samples)

target = Banana(2, 1.0f0, 100.0f0)
target_g = fmap(cu, target) # move target to GPU
logp_g = Base.Fix1(logpdf, target_g)

logp_g(yy)

