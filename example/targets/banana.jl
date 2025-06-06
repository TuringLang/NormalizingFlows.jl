"""
    Banana{T<:Real}

Multidimensional banana-shape distribution.

# Fields 
$(FIELDS)

# Explanation

The banana distribution is obtained by applying a transformation ϕ to a multivariate normal 
distribution ``\\mathcal{N}(0, \\text{diag}(var, 1, 1, …, 1))``. The transformation ϕ is defined as
```math
\\phi(x_1, … , x_p) = (x_1, x_2 - B x_1^² + \\text{var}*B, x_3, … , x_p)
```
which has a unit Jacobian determinant.

Hence the density "fb" of a p-dimensional banana distribution is given by
```math
fb(x_1, \\dots, x_p) = \\exp\\left[ -\\frac{1}{2}\\frac{x_1^2}{\\text{var}} -
\\frac{1}{2}(x_2 + B x_1^2 - \\text{var}*B)^2 - \\frac{1}{2}(x_3^2 + x_4^2 + \\dots
+ x_p^2) \\right] / Z,
```
where "B" is the "banananicity" constant, determining the curvature of a banana, and   
``Z = \\sqrt{\\text{var} * (2\\pi)^p)}`` is the normalization constant.

# Reference

Gareth O. Roberts and Jeffrey S. Rosenthal
"Examples of Adaptive MCMC."
Journal of computational and graphical statistics, Volume 18, Number 2 (2009): 349-367.
"""
struct Banana{T<:Real} <: ContinuousMultivariateDistribution
    "Dimension of the distribution, must be >= 2"
    dim::Int      # Dimension
    "Banananicity constant, the larger |b| the more curved the banana"
    b::T          # Curvature
    "Variance of the first dimension, must be > 0"
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
