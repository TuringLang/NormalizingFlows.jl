using Functors
using Bijectors
using Bijectors: with_logabsdet_jacobian, LeakyReLU, transform, inverse
using Flux: glorot_uniform

struct InvertibleMLP{T1,T2} <: Bijectors.Bijector
    shift::T1
    scale::T2
    invertible_activation::Bijectors.Bijector
end

@functor InvertibleMLP

function InvertibleMLP(dims::Int)
    return InvertibleMLP(glorot_uniform(dims), glorot_uniform(dims), ParamLeakyReLU(dims))
end

function Bijectors.transform(mlp::InvertibleMLP, xs::AbstractMatrix)
    scale = abs.(mlp.scale) .+ one.(mlp.scale) ./ 10^3
    xxs = scale .* xs .+ mlp.shift
    return mlp.invertible_activation(xxs)
end

function Bijectors.with_logabsdet_jacobian(mlp::InvertibleMLP, xs::AbstractMatrix)
    scale = abs.(mlp.scale) .+ one.(mlp.scale) ./ 10^3
    xxs = scale .* xs .+ mlp.shift
    logjacs_affine = sum(log ∘ abs, scale) * ones(eltype(xs), size(xs, 2))
    ys, logjacs_act = with_logabsdet_jacobian(mlp.invertible_activation, xxs)
    return ys, logjacs_affine .+ logjacs_act
end

function Bijectors.with_logabsdet_jacobian(
    imlp::Inverse{<:InvertibleMLP}, ys::AbstractMatrix
)
    mlp = imlp.orig
    scale = abs.(mlp.scale) .+ one.(mlp.scale) ./ 10^3
    iact = inverse(mlp.invertible_activation)
    xxs, logjacs_act = with_logabsdet_jacobian(iact, ys)
    xs = (xxs .- mlp.shift) ./ scale
    logjacs_affine = -sum(log ∘ abs, scale) * ones(eltype(xs), size(xs, 2))
    logjacs = logjacs_affine .+ logjacs_act
    return xs, logjacs
end
function Bijectors.transform(imlp::Inverse{<:InvertibleMLP}, ys::AbstractVector)
    return Bijectors.with_logabsdet_jacobian(imlp, ys)[1]
end

struct ParamLeakyReLU{T} <: Bijectors.Bijector
    α::T
end

Functors.@functor ParamLeakyReLU

# Constructor with default alpha value
ParamLeakyReLU() = ParamLeakyReLU(0.5f0)
ParamLeakyReLU(dim::Int) = ParamLeakyReLU(0.5f0 * ones(Float32, dim))
ParamLeakyReLU(α) = Bijectors.LeakyReLU(abs.(α) .+ 1.0f-2)

# Inverse transformation method
Bijectors.inverse(b::ParamLeakyReLU) = Bijectors.LeakyReLU(inv.(abs.(b.α) .+ 1.0f-2))

# Forward transformation method
function Bijectors.transform(b::LeakyReLU, x::AbstractArray{<:Real})
    mask = x .< zero(eltype(x))
    J = mask .* b.α .+ (!).(mask)
    return J .* x
end

function Bijectors.with_logabsdet_jacobian(b::LeakyReLU, xs::AbstractMatrix{<:Real})
    mask = xs .< zero(eltype(xs))
    J = mask .* b.α .+ (!).(mask)
    return J .* xs, map(x_ -> sum(log.(abs.(x_))), eachcol(J))
end
