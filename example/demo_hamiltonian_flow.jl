using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Mooncake
using Bijectors 
using Bijectors: partition, combine, PartitionMask

using NormalizingFlows

include("SyntheticTargets.jl")
include("utils.jl")


"""
Hamiltonian flow interleaves betweem Leapfrog layer and an affine transformation to the momentum variable
It targets the joint distribution π(x, ρ) = π(x) * N(ρ; 0, I)
where x is the target variable and ρ is the momentum variable

- The optimizable parameters for the leapfrog layers are the step size logϵ
- and we will also optimize the affine transformation parameters (shift and scale) 

Wrap the leapfrog transformation into a Bijectors.jl interface

# References
[1] Naitong Chen, Zuheng Xu, Trevor Campbell, Bayesian inference via sparse Hamiltonian flows, NeurIPS 2022.
"""
struct LeapFrog{T<:Real} <: Bijectors.Bijector
    "dimention of the target space"
    dim::Int
    "leapfrog step size"
    logϵ::AbstractVector{T}
    "number of leapfrog steps"
    L::Int
    "score function of the target distribution"
    ∇logp
    "mask function to split the input into position and momentum"
    mask::PartitionMask
end
@functor LeapFrog (logϵ,)

function LeapFrog(dim::Int, logϵ::T, L::Int, ∇logp) where {T<:Real}
    return LeapFrog(dim, logϵ .* ones(T, dim), L, ∇logp, PartitionMask(2dim, 1:dim))
end

_get_stepsize(lf::LeapFrog) = exp.(lf.logϵ)

"""
run L leapfrog steps with std Gaussian momentum distribution with vector stepsizes
"""
function _leapfrog(
    ∇ℓπ, ϵ::AbstractVector{T}, L::Int, x::AbstractVecOrMat{T}, v::AbstractVecOrMat{T}
) where {T<:Real}
    v += ϵ/2 .* ∇ℓπ(x) 
    for _ in 1:L - 1
        x += ϵ .* v
        v += ϵ .* ∇ℓπ(x)
    end
    x += ϵ .* v
    v += ϵ/2 .* ∇ℓπ(x)
    return x, v
end

function Bijectors.transform(lf::LeapFrog{T}, z::AbstractVector{T}) where {T<:Real}
    (; dim, logϵ, L, ∇logp) = lf
    @assert length(z) == 2dim "dimension of input must be even, z = [x, ρ]"

    ϵ = _get_stepsize(lf) 
    x, ρ, e = partition(lf.mask, z)         # split into position and momentum
    x_, ρ_ = _leapfrog(∇logp, ϵ, L, x, ρ)   # run L learpfrog steps
    return combine(lf.mask, x_, ρ_, e)
end

function Bijectors.transform(ilf::Inverse{<:LeapFrog{T}}, z::AbstractVector{T}) where {T<:Real}
    lf = ilf.orig
    (; dim, logϵ, L, ∇logp) = lf
    @assert length(z) == 2dim "dimension of input must be even, z = [x, ρ]"

    ϵ = _get_stepsize(lf) 
    x, ρ, e = partition(lf.mask, z)         # split into position and momentum
    x_, ρ_ = _leapfrog(∇logp, -ϵ, L, x, ρ)   # run L learpfrog steps
    return combine(lf.mask, x_, ρ_, e)
end

function Bijectors.with_logabsdet_jacobian(lf::LeapFrog{T}, z::AbstractVector{T}) where {T<:Real}
    # leapfrog is symplectic, so the logabsdetjacobian is 0
    return Bijectors.transform(lf, z), zero(eltype(z))
end
function Bijectors.with_logabsdet_jacobian(ilf::Inverse{<:LeapFrog{T}}, z::AbstractVector{T}) where {T<:Real}
    # leapfrog is symplectic, so the logabsdetjacobian is 0
    return Bijectors.transform(ilf, z), zero(eltype(z))
end

# shift and scale transformation that only applies to the momentum variable ρ = z[(dim + 1):end]
function momentum_normalization_layer(dims::Int, T::Type{<:Real})
    bx = identity  # leave position variable x = z[1:dim] unchanged
    bρ = Bijectors.Shift(zeros(T, dims)) ∘ Bijectors.Scale(ones(T, dims))
    b = Bijectors.Stacked((bx, bρ), [1:dims, (dims + 1):(2*dims)])
    return b
end


##################################
# start demo
#################################
Random.seed!(123)
rng = Random.default_rng()
T = Float64 # for Hamiltonian VI, its recommended to use Float64 as the dynamic is chaotic

######################################
# a Funnel target
######################################
dims = 2
target = Funnel(dims, -8.0, 5.0)
# visualize(target)

logp = Base.Fix1(logpdf, target)
function logp_joint(z::AbstractVector{T}) where {T<:Real}
    dims = div(length(z), 2)
    x = @view z[1:dims]
    ρ = @view z[(dims + 1):end]
    logp_x = logp(x)
    logp_ρ = sum(logpdf(Normal(), ρ))
    return logp_x + logp_ρ
end

# the score function is the gradient of the logpdf. 
# In all the synthetic targets, the score function is only implemented for the Banana target
∇logp = Base.Fix1(score, target)

######################################
# build the flow in the joint space
######################################
# mean field Gaussian reference
@leaf MvNormal
q0 = transformed(
    MvNormal(zeros(T, 2dims), ones(T, 2dims)), Bijectors.Shift(zeros(T, 2dims)) ∘ Bijectors.Scale(ones(T, 2dims))
)

nlfg = 3
logϵ0 = log(0.05) # initial step size

# Hamiltonian flow interleaves betweem Leapfrog layer and an affine transformation to the momentum variable
Ls = [
    momentum_normalization_layer(dims, T) ∘ LeapFrog(dims, logϵ0, nlfg, ∇logp) for _ in 1:15
]

flow = create_flow(Ls, q0)
flow_untrained = deepcopy(flow)


######################################
# start training
######################################
sample_per_iter = 16

# callback function to log training progress
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,ad=adtype)
adtype = ADTypes.AutoMooncake(; config = Mooncake.Config())
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < one(T)/1000
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp_joint,
    sample_per_iter;
    max_iters=50_000,
    optimiser=Optimisers.Adam(3e-4),
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
