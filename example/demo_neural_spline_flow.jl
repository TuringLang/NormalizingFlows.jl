using Flux
using Functors
using Bijectors
using Bijectors: partition, combine, PartitionMask

using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Mooncake
using NormalizingFlows

include("common.jl")
include("SyntheticTargets.jl")
include("nn.jl")

##################################
# define neural spline layer using Bijectors.jl interface
#################################
"""
Neural Rational quadratic Spline layer 

# References
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). 
"""
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
    nn = [MLP_3layer(input_dims, hdims, 3K - 1) for _ in 1:num_of_transformed_dims]
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
hdims = 16
K = 10
B = 30
Ls = [
    NeuralSplineLayer(d, hdims, K, B, [1]) ∘ NeuralSplineLayer(d, hdims, K, B, [2]) for
    i in 1:3
]

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
    optimiser=Optimisers.Adam(2e-4),
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
