using Distributions, Random, LinearAlgebra, IrrationalConstants

"""
    WarpedGauss{T<:Real}

2-dimensional warped Gaussian distribution

# Fields 
$(FIELDS)

# Explanation

The banana distribution is obtained by applying a transformation ϕ to a 2-dimensional normal 
distribution ``\\mathcal{N}(0, diag(\\sigma_1, \\sigma_2))``. The transformation ϕ(x) is defined as
```math
ϕ(x_1, x_2) = (r*\cos(\theta + r/2), r*\sin(\theta + r/2)), 
```
where ``r = \\sqrt{x\_1^2 + x_2^2}``, ``\\theta = \\atan(x₂, x₁)``, 
and "atan(y, x) ∈ [-π, π]" is the angle, in radians, between the positive x axis and the 
ray to the point "(x, y)". See page 18. of [1] for reference.


# Reference
[1] Zuheng Xu, Naitong Chen, Trevor Campbell
"MixFlows: principled variational inference via mixed flows."
International Conference on Machine Learning, 2023
"""
struct WarpedGauss{T<:Real} <: ContinuousMultivariateDistribution
    "Standard deviation of the first dimension, must be > 0"
    σ1::T
    "Standard deviation of the second dimension, must be > 0"
    σ2::T
    function WarpedGauss{T}(σ1, σ2) where {T<:Real}
        σ1 > 0 || error("σ₁ must be > 0")
        σ2 > 0 || error("σ₂ must be > 0")
        return new{T}(σ1, σ2)
    end
end
WarpedGauss(σ1::T, σ2::T) where {T<:Real} = WarpedGauss{T}(σ1, σ2)
WarpedGauss() = WarpedGauss(1.0, 0.12)

Base.length(p::WarpedGauss) = 2
Base.eltype(p::WarpedGauss{T}) where {T<:Real} = T
Distributions.sampler(p::WarpedGauss) = p

# Define the transformation function φ and the inverse ϕ⁻¹ for the warped Gaussian distribution
function ϕ!(p::WarpedGauss, z::AbstractVector)
    length(z) == 2 || error("Dimension mismatch")
    x, y = z
    r = norm(z)
    θ = atan(y, x) #in [-π , π]
    θ -= r / 2
    z .= r .* [cos(θ), sin(θ)]
    return z
end

function ϕ⁻¹(p::WarpedGauss, z::AbstractVector)
    length(z) == 2 || error("Dimension mismatch")
    x, y = z
    r = norm(z)
    θ = atan(y, x) #in [-π , π]
    # increase θ depending on r to "smear"
    θ += r / 2

    # get the x,y coordinates foαtransformed point
    xn = r * cos(θ)
    yn = r * sin(θ)
    # compute jacobian
    logJ = log(r)
    return [xn, yn], logJ
end

function Distributions._rand!(rng::AbstractRNG, p::WarpedGauss, x::AbstractVecOrMat)
    size(x, 1) == 2 || error("Dimension mismatch")
    σ₁, σ₂ = p.σ₁, p.σ₂
    randn!(rng, x)
    x .*= [σ₁, σ₂]
    for y in eachcol(x)
        ϕ!(p, y)
    end
    return x
end

function Distributions._logpdf(p::WarpedGauss, x::AbstractVector)
    size(x, 1) == 2 || error("Dimension mismatch")
    σ₁, σ₂ = p.σ₁, p.σ₂
    S = [σ₁, σ₂] .^ 2
    z, logJ = ϕ⁻¹(p, x)
    return -sum(z .^ 2 ./ S) / 2 - IrrationalConstants.log2π - log(σ₁) - log(σ₂) + logJ
end
