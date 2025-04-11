using Functors
using Flux
using Bijectors
using Bijectors: partition, combine, PartitionMask
using SimpleUnPack: @unpack

abstract type TrainableScore end
struct CoresetScore{T<:AbstractVector} <: TrainableScore
    "coreset weights"
    w::T
    "weighted coreset score function of the target, ∇logpw(x, w)"
    ∇logpw
end
@functor CoresetScore (w,)
function CoresetScore(T, coresize::Int, datasize::Int, ∇logpw)
    return CoresetScore(ones(T, coresize) .* datasize ./ coresize, ∇logpw)
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

    # split into position and momentum
    # equivalent to x, ρ = z[1:dim], z[(dim + 1):end]
    mask = PartitionMask(n, 1:dim)
    x, ρ, _ = partition(mask, z) 

    # leapfrog step
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
