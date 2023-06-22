using Distributions, Bijectors, Optimisers
using LinearAlgebra
using Random
using NormalizingFlows
using Test

@testset "learining 2d Gaussian" begin
    μ = 10 * ones(Float32, 2)
    Σ = Diagonal(4 * ones(Float32, 2))
    target = MvNormal(μ, Σ)
    logp(z) = logpdf(target, z)

    q₀ = MvNormal(zeros(Float32, 2), ones(Float32, 2))
    flow = Bijectors.transformed(
        q₀, Bijectors.Shift(zero.(μ)) ∘ Bijectors.Scale(ones(Float32, 2))
    )

    sample_per_iter = 10
    flow_trained, stats, _ = NF(
        elbo,
        flow,
        logp,
        sample_per_iter;
        max_iters=5_000,
        optimiser=Optimisers.ADAM(1e-2),
        show_progress=false,
    )
    θ, re = Optimisers.destructure(flow_trained)

    el_untrained = elbo(Random.default_rng(), flow, logp, 1000)
    el_trained = elbo(flow_trained, logp, 1000)

    @test all(abs.(θ[1:2] .- μ) .< 0.2)
    @test all(abs.(θ[3:4] .- 2) .< 0.2)
    @test el_trained > el_untrained
    @test el_trained > -1
end