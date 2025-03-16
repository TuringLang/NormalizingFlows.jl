using Distributions, Random
# """
#     Cross(μ::Real=2.0, σ::Real=0.15)

# 2-dimensional Cross distribution

# # Explanation

# The Cross distribution is a 2-dimension 4-component Gaussian distribution with a "cross" 
# shape that is symmetric about the y- and x-axises. The mixture is defined as

# ```math
# \begin{aligned}
# p(x) =
# & 0.25 \mathcal{N}(x | (0, \mu), (\sigma, 1)) + \\
# & 0.25 \mathcal{N}(x | (\mu, 0), (1, \sigma)) + \\
# & 0.25 \mathcal{N}(x | (0, -\mu), (\sigma, 1)) + \\
# & 0.25 \mathcal{N}(x | (-\mu, 0), (1, \sigma)))
# \end{aligned}
# ```

# where ``μ`` and ``σ`` are the mean and standard deviation of the Gaussian components, 
# respectively. See an example of the Cross distribution in Page 18 of [1].

# # Reference
# [1] Zuheng Xu, Naitong Chen, Trevor Campbell
# "MixFlows: principled variational inference via mixed flows."
# International Conference on Machine Learning, 2023
# """
Cross() = Cross(2.0, 0.15)
function Cross(μ::T, σ::T) where {T<:Real}
    return MixtureModel([
        MvNormal([zero(μ), μ], [σ, one(σ)]),
        MvNormal([-μ, one(μ)], [one(σ), σ]),
        MvNormal([μ, one(μ)], [one(σ), σ]),
        MvNormal([zero(μ), -μ], [σ, one(σ)]),
    ])
end
function visualize(p::MixtureModel, samples=rand(p, 1000))
    xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1; length=100)
    yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1; length=100)
    z = [exp(Distributions.logpdf(p, [x, y])) for x in xrange, y in yrange]
    fig = contour(xrange, yrange, z'; levels=15, color=:viridis, label="PDF", linewidth=2)
    scatter!(samples[1, :], samples[2, :]; label="Samples", alpha=0.3, legend=:bottomright)
    return fig
end

function ∇pdf(dist::MvNormal, x::AbstractVector)
    return pdf(dist, x) * inv(dist.Σ) * (dist.μ .- x)
end

function ∇pdf(dist::MvNormal, xs::AbstractMatrix)
    return pdf(dist, xs)' .* (inv(dist.Σ) * (dist.μ .- xs))
end

function Score(p::MixtureModel, x::AbstractVector)
    pdfs = 4pdf(p, x)
    return mapreduce(dist -> ∇pdf(dist, x), +, p.components) ./ pdfs
end

function Score(p::MixtureModel, xs::AbstractMatrix)
    pdfs = 4 * pdf(p, xs)'
    return mapreduce(dist -> ∇pdf(dist, xs), +, p.components) ./ pdfs
end

# p = Cross()
# lp = Base.Fix1(logpdf, p)
# ∇l(x) = Zygote.gradient(lp, x)[1]

# dist = p.components[1]
# ∇ld(x) = Zygote.gradient(Base.Fix1(pdf, dist), x)[1]