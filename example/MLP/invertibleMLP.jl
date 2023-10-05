using Functors
using Bijectors
using Bijectors: LeakyReLU, with_logabsdet_jacobian
using Flux: glorot_uniform

struct InvertibleMLP{T1,T2} <: Bijectors.Bijector
    shift::T1
    scale::T2
    invertible_activation::Bijectors.Bijector
end

@functor InvertibleMLP

function InvertibleMLP(dims::Int)
    return InvertibleMLP(
        glorot_uniform(dims), glorot_uniform(dims), LeakyReLU(1.0f-2 * ones(dims))
    )
end

function Bijectors.transform(mlp::InvertibleMLP, xs::AbstractMatrix)
    xxs = mlp.scale .* xs .+ mlp.shift
    return mlp.invertible_activation(xxs)
end

function Bijectors.with_logabsdet_jacobian(mlp::InvertibleMLP, xs::AbstractMatrix)
    xxs = mlp.scale .* xs .+ mlp.shift
    logjacs_affine = sum(log ∘ abs, mlp.scale) * ones(size(xs, 2))
    ys, logjacs_act = with_logabsdet_jacobian(mlp.invertible_activation, xxs)
    return ys, logjacs_affine .+ logjacs_act
end

function Bijectors.with_logabsdet_jacobian(
    imlp::Inverse{<:InvertibleMLP}, ys::AbstractMatrix
)
    mlp = imlp.orig
    iact = inverse(mlp.invertible_activation)
    xxs, logjacs_act = with_logabsdet_jacobian(iact, ys)
    xs = (xxs .- mlp.shift) ./ mlp.scale
    logjacs_affine = -sum(log ∘ abs, mlp.scale) * ones(size(xs, 2))
    logjacs = logjacs_affine .+ logjacs_act
    return xs, logjacs
end
function Bijectors.transform(imlp::Inverse{<:InvertibleMLP}, ys::AbstractVector)
    return Bijectors.with_logabsdet_jacobian(imlp, ys)[1]
end

lu = Bijectors.LeakyReLU(0.01 * ones(4))

function Bijectors.with_logabsdet_jacobian(b::LeakyReLU, xs::AbstractMatrix)
    mask = xs .< zero(eltype(xs))
    J = mask .* b.α .+ (!).(mask)
    return J .* xs, map(x_ -> sum(log.(abs.(x_))), eachcol(J))
end
