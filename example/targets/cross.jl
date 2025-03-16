using Distributions, Random
"""
    Cross(μ::Real=2.0, σ::Real=0.15)

2-dimensional Cross distribution


# Explanation

The Cross distribution is a 2-dimension 4-component Gaussian distribution with a "cross" 
shape that is symmetric about the y- and x-axises. The mixture is defined as

```math
\begin{aligned}
p(x) =
& 0.25 \mathcal{N}(x | (0, \mu), (\sigma, 1)) + \\
& 0.25 \mathcal{N}(x | (\mu, 0), (1, \sigma)) + \\
& 0.25 \mathcal{N}(x | (0, -\mu), (\sigma, 1)) + \\
& 0.25 \mathcal{N}(x | (-\mu, 0), (1, \sigma)))
\end{aligned}
```

where ``μ`` and ``σ`` are the mean and standard deviation of the Gaussian components, 
respectively. See an example of the Cross distribution in Page 18 of [1].

# Reference
[1] Zuheng Xu, Naitong Chen, Trevor Campbell
"MixFlows: principled variational inference via mixed flows."
International Conference on Machine Learning, 2023
"""
Cross() = Cross(2.0, 0.15)
function Cross(μ::T, σ::T) where {T<:Real}
    return MixtureModel([
        MvNormal([zero(μ), μ], [σ, one(σ)]),
        MvNormal([-μ, one(μ)], [one(σ), σ]),
        MvNormal([μ, one(μ)], [one(σ), σ]),
        MvNormal([zero(μ), -μ], [σ, one(σ)]),
    ])
end
