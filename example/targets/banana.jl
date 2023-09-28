using Distributions, Random
using Plots
using IrrationalConstants

"""
    Banana{T<:Real}

Multidimensional banana-shape distribution.

# Fields 
- 'dim::Int': Dimension of the distribution, must be >= 2
- 'b::T': Banananicity constant, the larger "|b|" the more curved the banana
- 'var::T': Variance of the first dimension, must be > 0


# Explanation

The banana distribution is obtained by applying a transformation ϕ to a multivariate normal 
distribution "N(0, diag(var, 1, 1, …, 1))". The transformation ϕ is defined as
"ϕ(x₁, … , xₚ) = (x₁, x₂ - B x₁² + var*B, x₃, … , xₚ)", 
which has a unit Jacobian determinant.

Hence the density "fb" of a p-dimensional banana distribution is given by
"fb(x₁, … , xₚ) = exp[ -½x₁²/var - ½(x₂ + B x₁² - var*B)² - ½(x₃² + x₄² + … + xₚ²)] / Z",
where "B" is the "banananicity" constant, determining the curvature of a banana, and   
"Z = sqrt(var * (2π)^p))" is the normalization constant.


# Reference

Gareth O. Roberts and Jeffrey S. Rosenthal
"Examples of Adaptive MCMC."
Journal of computational and graphical statistics, Volume 18, Number 2 (2009): 349-367.
"""
struct Banana{T<:Real} <: ContinuousMultivariateDistribution
    dim::Int      # Dimension
    b::T          # Curvature
    var::T        # Variance
    function Banana{T}(dim::Int, b::T, var::T) where {T<:Real}
        dim >= 2 || error("dim must be >= 2")
        var > 0 || error("var must be > 0")
        return new{T}(dim, b, var)
    end
end
Banana(dim::Int, b::T, var::T) where {T<:Real} = Banana{T}(dim, b, var)

Base.length(p::Banana) = p.dim
Base.eltype(p::Banana{T}) where {T<:Real} = T
Distributions.sampler(p::Banana) = p

# Define the transformation function φ and the inverse ϕ⁻¹ for the banana distribution
function ϕ!(p::Banana, x::AbstractVector)
    d, b, s = p.dim, p.b, p.var
    d == length(x) || error("Dimension mismatch")
    return x[2] = x[2] - b * x[1]^2 + s * b
end
function ϕ⁻¹(p::Banana, x::AbstractVector)
    d, b, s = p.dim, p.b, p.var
    d == length(x) || error("Dimension mismatch")
    y2 = x[2] + b * x[1]^2 - s * b
    return d == 2 ? [x[1], y2] : reduce(vcat, [x[1], y2, x[3:end]])
end

function Distributions._rand!(rng::AbstractRNG, p::Banana, x::AbstractVecOrMat)
    T = eltype(p)
    d, s = p.dim, p.var
    d == size(x, 1) || error("Dimension mismatch")
    x[1, :] .= randn(rng, T, size(x, 2)) .* sqrt(s)
    x[2:end, :] .= randn(rng, T, d - 1, size(x, 2))
    for y in eachcol(x)
        ϕ!(p, y)
    end
    return x
end

function Distributions._logpdf(p::Banana, x::AbstractVector)
    T = eltype(p)
    d, b, s = p.dim, p.b, p.var
    ϕ⁻¹_x = ϕ⁻¹(p, x)
    logz = (log(s) / d + IrrationalConstants.log2π) * d / 2
    return -logz - sum(ϕ⁻¹_x .^ 2 ./ vcat(s, ones(T, d - 1))) / 2
end

function Distributions._logpdf(p::Banana, xs::AbstractMatrix)
    return map(_logpdf(p), eachcol(xs))
end

function visualize(p::Banana, samples=rand(p, 1000))
    xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1; length=100)
    yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1; length=100)
    z = [exp(Distributions.logpdf(p, [x, y])) for x in xrange, y in yrange]
    fig = contour(xrange, yrange, z'; levels=15, color=:viridis, label="PDF", linewidth=2)
    scatter!(samples[1, :], samples[2, :]; label="Samples", alpha=0.3, legend=:bottomright)
    return fig
end
