using Distributions
using Plots

struct Banana{T<:Real} <: ContinuousMultivariateDistribution
    d::Int            # Dimension
    b::T        # Curvature
    Z::T        # Normalizing constant
    C::Matrix{T} # Covariance matrix
    C⁻¹::Matrix{T} # Inverse of covariance matrix
end

# Constructor with additional scaling parameter s
function Banana(d::Int, b::T, s::T=100.0f0) where {T<:Real}
    return Banana(
        d,
        b,
        T(sqrt(s * (2π)^d)),
        Matrix(Diagonal(vcat(s, ones(T, d - 1)))),
        Matrix(Diagonal(vcat(1 / s, ones(T, d - 1)))),
    )
end
Base.length(p::Banana) = p.d

Distributions.sampler(p::Banana) = p

# Define the transformation function φ and the inverse ϕ⁻¹ for the banana distribution
φ(x, b, s) = [x[1], x[2] + b * x[1]^2 - s * b]
ϕ⁻¹(y, b, s) = [y[1], y[2] - b * y[1]^2 + s * b]

function Distributions._rand!(rng::AbstractRNG, p::Banana, x::AbstractArray{<:Real})
    b, C = p.b, p.C
    mvnormal = MvNormal(zeros(2), C)
    for i in axes(x, 2)
        x[:, i] = φ(rand(rng, mvnormal), b, C[1, 1])
    end
    return x
end

function Distributions._logpdf(p::Banana, x::AbstractArray)
    Z, C⁻¹, b = p.Z, p.C⁻¹, p.b
    ϕ⁻¹_x = ϕ⁻¹(x, b, p.C[1, 1])
    return -log(Z) - ϕ⁻¹_x' * C⁻¹ * ϕ⁻¹_x / 2
end

function visualize(p::Banana, samples=rand(p, 1000))
    xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1; length=100)
    yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1; length=100)
    z = [exp(Distributions.logpdf(p, [x, y])) for x in xrange, y in yrange]
    contour(xrange, yrange, z'; levels=15, color=:viridis, label="PDF", linewidth=2)
    scatter!(samples[1, :], samples[2, :]; label="Samples", alpha=0.3, legend=:bottomright)
    return current()
end