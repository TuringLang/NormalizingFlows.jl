using Distributions, Bijectors, Optimisers
using LinearAlgebra
using Random
using NormalizingFlows
using Test

@testset "variational objectives" begin
    μ = randn(Float32, 2)
    Σ = Diagonal(rand(Float32, 2) .+ 1.0f-3)
    target = MvNormal(μ, Σ)
    logp(z) = logpdf(target, z)

    q₀ = MvNormal(zeros(Float32, 2), ones(Float32, 2))
    flow = Bijectors.transformed(q₀, Bijectors.Shift(μ) ∘ Bijectors.Scale(sqrt.(Σ)))

    x = randn(Float32, 2)

    @testset "elbo" begin
        el = elbo(Random.default_rng(), flow, logp, 10)

        @test abs(el) < 1.0f-5
        @test logpdf(flow, x) + el ≈ logp(x)
    end

    @testset "likelihood" begin
        sample_trained = rand(flow, 1000)
        sample_untrained = rand(q₀, 1000)
        llh_trained = loglikelihood(flow, sample_trained)
        llh_untrained = loglikelihood(flow, sample_untrained)

        @test llh_trained > llh_untrained
    end
end
