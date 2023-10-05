using Functors
using Bijectors

struct InvertibleMLP{T1,T2} <: Bijectors.Bijector
    shift::T1
    scale::T2
    invertible_activation::Bijectors.Bijector
end

@functor InvertibleMLP

function Bijectors.with_logabsdet_jacobian(mlp::InvertibleMLP, xs::AbstractMatrix)
    ys = mlp.scale .* (xs .- mlp.shift)
    logjacs = sum(log ∘ abs, mlp.scale) * ones(size(xs, 2))
    return ys, logjacs
end
