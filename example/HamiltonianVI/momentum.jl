using Statistics: mean, std
using Functors
using Bijectors

mutable struct MomentumNorm <: Bijectors.Bijector
    dim::Int
    shift
    scale
    inv_scale
end

@functor MomentumNorm (shift, scale, inv_scale)

MomentumNorm(dim::Int) = MomentumNorm(dim, nothing, nothing, nothing)

function Bijectors.transform(af::MomentumNorm, zs::AbstractVecOrMat)
    return Bijectors.with_logabsdet_jacobian(af, zs)[1]
end
function Bijectors.transform(b::Inverse{<:MomentumNorm}, zs::AbstractVecOrMat)
    return Bijectors.with_logabsdet_jacobian(b, zs)[1]
end

# Xs = 2 .* randn(2, 1000) .+ 10
# Σ = cov(Xs')
# C = cholesky(Hermitian(Σ))
# Ys = inv(C.L) * (Xs .- mean(Xs; dims=2))

function Bijectors.with_logabsdet_jacobian(af::MomentumNorm, zs::AbstractMatrix)
    dim = af.dim
    # split into position and momentum
    xs, ρs = zs[1:dim, :], zs[(dim + 1):end, :]
    if af.shift === nothing && af.scale === nothing
        af.shift = vec(mean(ρs; dims=2))
        Σ = cov(ρs')
        C = cholesky(Hermitian(Σ))
        af.inv_scale = C.L
        af.scale = inv(C.L)
    end
    ρns = af.scale * (ρs .- af.shift)
    ys = vcat(xs, ρns)
    logjacs = sum(log ∘ abs, diag(af.scale)) * ones(size(zs, 2))
    return ys, logjacs
end

function Bijectors.with_logabsdet_jacobian(af::MomentumNorm, z::AbstractVector)
    dim = af.dim
    # split into position and momentum
    x, ρ = z[1:dim], z[(dim + 1):end]
    @assert af.shift !== nothing && af.scale !== nothing "shift and scale must be specified"
    ρn = af.scale * (ρ .- af.shift)
    y = vcat(x, ρn)
    logjac = sum(log ∘ abs, diag(af.scale))
    return y, logjac
end

function Bijectors.with_logabsdet_jacobian(iaf::Inverse{<:MomentumNorm}, ys::AbstractMatrix)
    af = iaf.orig
    dim = af.dim
    # split into position and momentum
    xs, ρs = ys[1:dim, :], ys[(dim + 1):end, :]
    if af.shift === nothing && af.scale === nothing
        throw(ArgumentError("shift and scale must be specified"))
    end
    ρns = af.inv_scale * ρs .+ af.shift
    logjacs = sum(log ∘ abs, diag(af.scale)) * ones(size(ys, 2))
    return vcat(xs, ρns), -logjacs
end

function Bijectors.with_logabsdet_jacobian(iaf::Inverse{<:MomentumNorm}, y::AbstractVector)
    af = iaf.orig
    dim = af.dim
    # split into position and momentum
    x, ρ = y[1:dim], y[(dim + 1):end]
    if af.shift === nothing && af.scale === nothing
        throw(ArgumentError("shift and scale must be specified"))
    end
    ρn = af.inv_scale * ρ .+ af.shift
    logjac = sum(log ∘ abs, diag(af.scale))
    return vcat(x, ρn), -logjac
end
