using Functors
using Bijectors
using Bijectors: partition, combine, PartitionMask
using SimpleUnPack: @unpack

struct LeapFrog{T<:Real,I<:Int} <: Bijectors.Bijector
    "dimention of the target space"
    dim::I
    "leapfrog step size"
    ϵ::AbstractVector{T}
    "number of leapfrog steps"
    L::I
    "score of the target distribution"
    ∇logp
    "score of the momentum distribution"
    ∇logm
end
@functor LeapFrog (ϵ,)

function LeapFrog(dim::Int, ϵ::AbstractVector{T}, L::Int, ∇logp, ∇logm)
    return LeapFrog(dim, ϵ .* one.(ϵ), L, ∇logp, ∇logm)
end

function Bijectors.inverse(lf::LeapFrog)
    @unpack d, ϵ, L, ∇logp, ∇logm = lf
    return LeapFrog(d, -ϵ, L, ∇logp, ∇logm)
end

function Bijectors.transform(lf::LeapFrog, z::AbstractVector)
    @unpack d, ϵ, L, ∇logp, ∇logm = lf
    @assert length(z) == 2d "dimension of input must be even, z = [x, ρ]"
    mask = PartitionMask(n, 1:(lf.dim))
    x, ρ, emp = partition(mask, z)

    ρ += ϵ / 2 .* ∇logp(x)
    for i in 1:(lf.L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* ∇logp(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ / 2 .* ∇logp(x)
    return combine(mask, x, ρ, emp)
end

# leapfrog composes shear transformations, hence has unit jacobian 
Bijector.logabsdetjac(lf::LeapFrog, z::AbstractVector) = zero(eltype(z))

function Bijectors.with_logabsdet_jacobian(lf::LeapFrog, z::AbstractVector)
    return Bijectors.transform(lf, z), zero(eltype(z))
end

abstract type TrainableScore end
struct CoresetScore{T<:AbstractVector} <: TrainableScore
    "coreset weights"
    w::T
    "weighted coreset score function of the target, ∇logpw(w, x)"
    ∇logpw
end

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

function SurrogateLeapFrog(dim::Int, ϵ::AbstractVector{T}, L::Int, ∇S, ∇logm)
    return SurrogateLeapFrog(dim, ϵ .* one.(ϵ), L, ∇S, ∇logm)
end

function Bijectors.inverse(slf::SurrogateLeapFrogeapFrog)
    @unpack d, ϵ, L, ∇S, ∇logm = slf
    return SurrogateLeapFrog(d, -ϵ, L, ∇S, ∇logm)
end

function Bijectors.transform(slf::SurrogateLeapFrog, z::AbstractVector)
    @unpack d, ϵ, L, ∇S, ∇logm = slf
    @assert length(z) == 2d "dimension of input must be even, z = [x, ρ]"
    mask = PartitionMask(n, 1:(lf.dim))
    x, ρ, emp = partition(mask, z)

    ρ += ϵ / 2 .* ∇S(x)
    for i in 1:(lf.L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* ∇S(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ / 2 .* ∇S(x)
    return combine(mask, x, ρ, emp)
end

# leapfrog composes shear transformations, hence has unit jacobian 
Bijector.logabsdetjac(slf::SurrogateLeapFrog, z::AbstractVector) = zero(eltype(z))

function Bijectors.with_logabsdet_jacobian(slf::SurrogateLeapFrog, z::AbstractVector)
    return Bijectors.transform(slf, z), zero(eltype(z))
end
