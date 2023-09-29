using Functors
using Flux
using Bijectors
using Bijectors: partition, combine, PartitionMask
using SimpleUnPack: @unpack

struct LeapFrog{T<:Real,I<:Int} <: Bijectors.Bijector
    "dimention of the target space"
    dim::I
    "leapfrog step size"
    logϵ::AbstractVector{T}
    "tempering"
    γ::AbstractVector{T}
    "number of leapfrog steps"
    L::I
    "score of the target distribution"
    ∇logp
    "score of the momentum distribution"
    ∇logm
end
@functor LeapFrog (logϵ, γ)

function LeapFrog(dim::Int, logϵ::T, γ::T, L::Int, ∇logp, ∇logm) where {T<:Real}
    return LeapFrog(dim, logϵ .* ones(T, dim), γ .* ones(T, dims), L, ∇logp, ∇logm)
end

# function Bijectors.inverse(lf::LeapFrog)
#     @unpack d, ϵ, L, ∇logp, ∇logm = lf
#     return LeapFrog(d, -ϵ, L, ∇logp, ∇logm)
# end

function Bijectors.transform(lf::LeapFrog, z::AbstractVector)
    @unpack dim, logϵ, γ, L, ∇logp, ∇logm = lf
    ϵ = exp.(logϵ)
    @assert length(z) == 2dim "dimension of input must be even, z = [x, ρ]"
    # mask = PartitionMask(n, 1:dim)
    # x, ρ, emp = partition(mask, z)
    x, ρ = z[1:dim], z[(dim + 1):end]

    ρ += ϵ ./ 2 .* γ .* ∇logp(x)
    for i in 1:(L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* γ .* ∇logp(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ ./ 2 .* γ .* ∇logp(x)
    # return combine(mask, x, ρ, emp)
    return vcat(x, ρ)
end

function Bijectors.transform(ilf::Inverse{<:LeapFrog}, z::AbstractVector)
    lf = ilf.orig
    @unpack dim, logϵ, γ, L, ∇logp, ∇logm = lf
    ϵ = -exp.(logϵ) # flip momentum sign
    @assert length(z) == 2dim "dimension of input must be even, z = [x, ρ]"
    # mask = PartitionMask(n, 1:dim)
    # x, ρ, emp = partition(mask, z)
    x, ρ = z[1:dim], z[(dim + 1):end]

    ρ += ϵ ./ 2 .* γ .* ∇logp(x)
    for i in 1:(L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* γ .* ∇logp(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ ./ 2 .* γ .* ∇logp(x)
    # return combine(mask, x, ρ, emp)
    return vcat(x, ρ)
end

function Bijectors.with_logabsdet_jacobian(lf::LeapFrog, z::AbstractVector)
    return Bijectors.transform(lf, z), zero(eltype(z))
end
function Bijectors.with_logabsdet_jacobian(ilf::Inverse{<:LeapFrog}, z::AbstractVector)
    return Bijectors.transform(ilf, z), zero(eltype(z))
end

abstract type TrainableScore end
struct CoresetScore{T<:AbstractVector} <: TrainableScore
    "coreset weights"
    w::T
    "weighted coreset score function of the target, ∇logpw(x, w)"
    ∇logpw
end
@functor CoresetScore (w,)
function CoresetScore(T, coresize::Int, datasize::Int, ∇logpw)
    return CoresetScore(ones(T, coresize) .* N ./ coresize, ∇logpw)
end
(C::CoresetScore)(x::AbstractVector) = C.∇logpw(x, C.w)

struct SurrogateLeapFrog{T<:Real,I<:Int,H<:Union{TrainableScore,Flux.Chain}} <:
       Bijectors.Bijector
    "dimention of the target space"
    dim::I
    "leapfrog step size"
    ϵ::AbstractVector{T}
    "number of leapfrog steps"
    L::I
    "trainable surrogate of the score of the target distribution, e.g., coreset score or some neural net"
    ∇S::H
    "score of the momentum distribution"
    ∇logm
end

@functor SurrogateLeapFrog (ϵ, ∇S)

function SurrogateLeapFrog(dim::Int, ϵ::T, L::Int, ∇S, ∇logm) where {T<:Real}
    return SurrogateLeapFrog(dim, ϵ .* ones(T, dims), L, ∇S, ∇logm)
end

function Bijectors.inverse(slf::SurrogateLeapFrog)
    @unpack dim, ϵ, L, ∇S, ∇logm = slf
    return SurrogateLeapFrog(dim, -ϵ, L, ∇S, ∇logm)
end

function Bijectors.transform(slf::SurrogateLeapFrog, z::AbstractVector)
    @unpack dim, ϵ, L, ∇S, ∇logm = slf
    n = length(z)
    @assert n == 2dim "dimension of input must be even, z = [x, ρ]"
    # mask = PartitionMask(n, 1:dim)
    x, ρ = z[1:dim], z[(dim + 1):end]
    # x, ρ, emp = partition(mask, z)

    ρ += ϵ ./ 2 .* ∇S(x)
    for i in 1:(L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* ∇S(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ ./ 2 .* ∇S(x)
    # return combine(mask, x, ρ, emp)
    return vcat(x, ρ)
end

# leapfrog composes shear transformations, hence has unit jacobian 
function Bijectors.with_logabsdet_jacobian(slf::SurrogateLeapFrog, z::AbstractVector)
    return Bijectors.transform(slf, z), zero(eltype(z))
end
