"""
    Funnel{T<:Real}

Multidimensional Neal's Funnel distribution

# Fields 
$(FIELDS)

# Explanation

The Neal's Funnel distribution is a p-dimensional distribution with a funnel shape, 
originally proposed by Radford Neal in [2]. 
The marginal distribution of ``x_1`` is Gaussian with mean "μ" and standard
deviation "σ". The conditional distribution of ``x_2, \\dots, x_p | x_1`` are independent 
Gaussian distributions with mean 0 and standard deviation ``\\exp(x_1/2)``. 
The generative process is given by
```math
x_1 \\sim \\mathcal{N}(\\mu, \\sigma^2), \\quad x_2, \\ldots, x_p \\sim \\mathcal{N}(0, \\exp(x_1))
```

# Reference
[1] Stan User’s Guide: 
https://mc-stan.org/docs/2_18/stan-users-guide/reparameterization-section.html#ref-Neal:2003
[2] Radford Neal 2003. “Slice Sampling.” Annals of Statistics 31 (3): 705–67.
"""
struct Funnel{T<:Real} <: ContinuousMultivariateDistribution
    "Dimension of the distribution, must be >= 2"
    dim::Int
    "Mean of the first dimension"
    μ::T
    "Standard deviation of the first dimension, must be > 0"
    σ::T
    function Funnel{T}(dim::Int, μ::T, σ::T) where {T<:Real}
        dim >= 2 || error("dim must be >= 2")
        σ > 0 || error("σ must be > 0")
        return new{T}(dim, μ, σ)
    end
end
Funnel(dim::Int, μ::T, σ::T) where {T<:Real} = Funnel{T}(dim, μ, σ)
Funnel(dim::Int, σ::T) where {T<:Real} = Funnel{T}(dim, zero(T), σ)
Funnel(dim::Int) = Funnel(dim, 0.0, 9.0)

Base.length(p::Funnel) = p.dim
Base.eltype(p::Funnel{T}) where {T<:Real} = T

function Distributions._rand!(rng::AbstractRNG, p::Funnel{T}, x::AbstractVecOrMat{T}) where {T<:Real}
    d, μ, σ = p.dim, p.μ, p.σ
    d == size(x, 1) || error("Dimension mismatch")
    x[1, :] .= randn(rng, T, size(x, 2)) .* σ .+ μ
    x[2:end, :] .= randn(rng, T, d - 1, size(x, 2)) .* exp.(@view(x[1, :]) ./ 2)'
    return x
end

function Distributions._logpdf(p::Funnel{T}, x::AbstractVector{T}) where {T<:Real}
    d, μ, σ = p.dim, p.μ, p.σ
    x1 = x[1]
    x2 = x[2:end]
    lpdf_x1 = logpdf(Normal(μ, σ), x1)
    lpdf_x2_given_1 = logpdf(MvNormal(zeros(T, d-1), exp(x1)I), x2)
    return lpdf_x1 + lpdf_x2_given_1
end

function score(p::Funnel{T}, x::AbstractVector{T}) where {T<:Real}
    d, μ, σ = p.dim, p.μ, p.σ
    x1 = x[1]
    x_2_d = x[2:end]
    a = expm1(-x1) + 1

    ∇lpdf1 = (μ - x1)/σ^2 - (d-1)/2 + a*sum(abs2, x_2_d)/2
    ∇lpdfs = -a*x_2_d
    return vcat(∇lpdf1, ∇lpdfs)
end
