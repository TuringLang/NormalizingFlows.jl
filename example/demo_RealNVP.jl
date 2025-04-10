using Flux
using Functors
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

function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
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
    logjac = sum(log ∘ abs, af.s(x_2))
    return combine(af.mask, y_1, x_2, x_3), logjac
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

function Bijectors.logabsdetjac(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = partition(af.mask, x)
    logjac = sum(log ∘ abs, af.s(x_2))
    return logjac
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



##################################
# start demo
#################################
Random.seed!(123)
rng = Random.default_rng()
T = Float32

######################################
# a difficult banana target
######################################
target = Banana(2, 1.0f0, 100.0f0)
logp = Base.Fix1(logpdf, target)

######################################
# learn the target using Affine coupling flow
######################################
@leaf MvNormal
q0 = MvNormal(zeros(T, 2), ones(T, 2))

d = 2
hdims = 32
Ls = [AffineCoupling(d, hdims, [1]) ∘ AffineCoupling(d, hdims, [2]) for i in 1:3]

flow = create_flow(Ls, q0)
flow_untrained = deepcopy(flow)


######################################
# start training
######################################
sample_per_iter = 64

# callback function to log training progress
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
adtype = ADTypes.AutoMooncake(; config = Mooncake.Config())
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp,
    sample_per_iter;
    max_iters=50_000,
    optimiser=Optimisers.Adam(5e-4),
    ADbackend=adtype,
    show_progress=true,
    callback=cb,
    hasconverged=checkconv,
)
θ, re = Optimisers.destructure(flow_trained)
losses = map(x -> x.loss, stats)

######################################
# evaluate trained flow
######################################
plot(losses; label="Loss", linewidth=2) # plot the loss
compare_trained_and_untrained_flow(flow_trained, flow_untrained, target, 1000)
