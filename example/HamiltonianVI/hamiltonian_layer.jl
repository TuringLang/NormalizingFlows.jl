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
    "number of leapfrog steps"
    L::I
    "score function of the target distribution"
    ∇logp
    "score function of the momentum distribution"
    ∇logm
end
@functor LeapFrog (logϵ,)

function LeapFrog(dim::Int, logϵ::T, L::Int, ∇logp, ∇logm) where {T<:Real}
    return LeapFrog(dim, logϵ .* ones(T, dim), L, ∇logp, ∇logm)
end

function _leapfrog(
    x::AbstractVector{T}, ρ::AbstractVector{T}, ϵ::AbstractVector{T}, L::Int, ∇logp, ∇logm
) where {T<:Real}
    # run L learpfrog steps
    ρ += ϵ ./ 2 .* ∇logp(x)
    for i in 1:(L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* ∇logp(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ ./ 2 .* ∇logp(x)
    
    return x, ρ
end

# function Bijectors.inverse(lf::LeapFrog)
#     @unpack d, ϵ, L, ∇logp, ∇logm = lf
#     return LeapFrog(d, -ϵ, L, ∇logp, ∇logm)
# end

function Bijectors.transform(lf::LeapFrog, z::AbstractVector)
    @unpack dim, logϵ, L, ∇logp, ∇logm = lf
    ϵ = exp.(logϵ)
    @assert length(z) == 2dim "dimension of input must be even, z = [x, ρ]"
    # split into position and momentum
    x, ρ = z[1:dim], z[(dim + 1):end]

    # run L learpfrog steps
    ρ += ϵ ./ 2 .* ∇logp(x)
    for i in 1:(L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* ∇logp(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ ./ 2 .* ∇logp(x)
    return vcat(x, ρ)
end

function Bijectors.transform(ilf::Inverse{<:LeapFrog}, z::AbstractVector)
    lf = ilf.orig
    @unpack dim, logϵ, L, ∇logp, ∇logm = lf
    ϵ = -exp.(logϵ) # flip momentum sign
    @assert length(z) == 2dim "dimension of input must be even, z = [x, ρ]"
    # split into position and momentum
    x, ρ = z[1:dim], z[(dim + 1):end]

    # run L learpfrog steps
    ρ += ϵ ./ 2 .* ∇logp(x)
    for i in 1:(L - 1)
        x -= ϵ .* ∇logm(ρ)
        ρ += ϵ .* ∇logp(x)
    end
    x -= ϵ .* ∇logm(ρ)
    ρ += ϵ ./ 2 .* ∇logp(x)
    return vcat(x, ρ)
end

function Bijectors.with_logabsdet_jacobian(lf::LeapFrog, z::AbstractVector)
    return Bijectors.transform(lf, z), zero(eltype(z))
end
function Bijectors.with_logabsdet_jacobian(ilf::Inverse{<:LeapFrog}, z::AbstractVector)
    return Bijectors.transform(ilf, z), zero(eltype(z))
end

# function Bijectors.transform(lf::LeapFrog, zs::AbstractMatrix)
#     @unpack dim, logϵ, L, ∇logp, ∇logm = lf
#     ϵ = exp.(logϵ)
#     @assert size(zs, 1) == 2dim "dimension of input must be even, zs = [x, ρ]"
#     # split into position and momentum
#     xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]

#     # run L learpfrog steps
#     ρs += ϵ ./ 2 .* ∇logp(xs)
#     for i in 1:(L - 1)
#         xs -= ϵ .* ∇logm(ρs)
#         ρs += ϵ .* ∇logp(xs)
#     end
#     xs -= ϵ .* ∇logm(ρs)
#     ρs += ϵ ./ 2 .* ∇logp(xs)
#     return vcat(xs, ρs)
# end

# function Bijectors.transform(ilf::Inverse{<:LeapFrog}, zs::AbstractMatrix)
#     lf = ilf.orig
#     @unpack dim, logϵ, L, ∇logp, ∇logm = lf
#     ϵ = -exp.(logϵ) # flip momentum sign
#     @assert size(zs, 1) == 2dim "dimension of input must be even, zs = [x, ρ]"
#     # split into position and momentum
#     xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]

#     # run L learpfrog steps
#     ρs += ϵ ./ 2 .* ∇logp(xs)
#     for i in 1:(L - 1)
#         xs -= ϵ .* ∇logm(ρs)
#         ρs += ϵ .* ∇logp(xs)
#     end
#     xs -= ϵ .* ∇logm(ρs)
#     ρs += ϵ ./ 2 .* ∇logp(xs)
#     return vcat(xs, ρs)
# end
# function Bijectors.with_logabsdet_jacobian(lf::LeapFrog, zs::AbstractMatrix)
#     @assert size(zs, 1) == 2 * lf.dim "dimension of input must be even, zs = [x, ρ]"
#     return Bijectors.transform(lf, zs), zeros(eltype(zs), size(zs, 2))
# end
# function Bijectors.with_logabsdet_jacobian(ilf::Inverse{<:LeapFrog}, zs::AbstractMatrix)
#     lf = ilf.orig
#     @assert size(zs, 1) == 2 * lf.dim "dimension of input must be even, zs = [x, ρ]"
#     return Bijectors.transform(ilf, zs), zeros(eltype(zs), size(zs, 2))
# end
