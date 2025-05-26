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
# define neural spline layer using Bijectors.jl interface
#################################
"""
Neural Rational quadratic Spline layer 

# References
[1] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G., Neural Spline Flows, CoRR, arXiv:1906.04032 [stat.ML],  (2019). 
"""
struct NeuralSplineLayer{T,A<:Flux.Chain} <: Bijectors.Bijector
    dim::Int                # dimension of input
    K::Int                  # number of knots
    n_dims_transferred::Int  # number of dimensions that are transformed
    nn::A   # networks that parmaterize the knots and derivatives
    B::T                    # bound of the knots
    mask::Bijectors.PartitionMask
end

function NeuralSplineLayer(
    dim::T1,                # dimension of input
    hdims::T1,              # dimension of hidden units for s and t
    K::T1,                  # number of knots
    B::T2,                  # bound of the knots
    mask_idx::AbstractVector{<:Int}, # index of dimensione that one wants to apply transformations on
) where {T1<:Int,T2<:Real}
    num_of_transformed_dims = length(mask_idx)
    input_dims = dim - num_of_transformed_dims
    
    # output dim of the NN
    output_dims = (3K - 1)*num_of_transformed_dims
    # one big mlp that outputs all the knots and derivatives for all the transformed dimensions
    nn = mlp3(input_dims, hdims, output_dims)

    mask = Bijectors.PartitionMask(dim, mask_idx)
    return NeuralSplineLayer(dim, K, num_of_transformed_dims, nn, B, mask)
end

@functor NeuralSplineLayer (nn,)

# define forward and inverse transformation
"""
Build a rational quadratic spline from the nn output
Bijectors.jl has implemented the inverse and logabsdetjac for rational quadratic spline

we just need to map the nn output to the knots and derivatives of the RQS
"""
function instantiate_rqs(nsl::NeuralSplineLayer, x::AbstractVector)
    K, B = nsl.K, nsl.B
    nnoutput = reshape(nsl.nn(x), nsl.n_dims_transferred, :)
    ws = @view nnoutput[:, 1:K]
    hs = @view nnoutput[:, (K + 1):(2K)]
    ds = @view nnoutput[:, (2K + 1):(3K - 1)]
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
    x_1, x_2, _ = Bijectors.partition(nsl.mask, x)
    rqs = instantiate_rqs(nsl, x_2)
    logjac = logabsdetjac(rqs, x_1)
    return logjac
end

function Bijectors.logabsdetjac(insl::Inverse{<:NeuralSplineLayer}, y::AbstractVector)
    nsl = insl.orig
    y1, y2, _ = partition(nsl.mask, y)
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
# neals funnel target
######################################
target = Funnel(2, 0.0f0, 9.0f0)
logp = Base.Fix1(logpdf, target)

######################################
# learn the target using Affine coupling flow
######################################
@leaf MvNormal
q0 = MvNormal(zeros(T, 2), ones(T, 2))

d = 2
hdims = 64
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
    max_iters=100,   # change to larger number of iterations (e.g., 50_000) for better results
    optimiser=Optimisers.Adam(5e-5),
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
