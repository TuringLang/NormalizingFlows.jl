@testset "variational objectives" begin
    @testset "$T" for T in [Float32, Float64]
        μ = randn(T, 2)
        Σ = Diagonal(rand(T, 2) .+ T(1e-3))
        target = MvNormal(μ, Σ)
        logp(z) = logpdf(target, z)

        q₀ = MvNormal(zeros(T, 2), I)
        flow = Bijectors.transformed(q₀, Bijectors.Shift(μ) ∘ Bijectors.Scale(sqrt.(Σ)))

        x = randn(T, 2)
        rng = Random.default_rng()

        @testset "elbo" begin
            el = elbo(rng, flow, logp, 10)

            @test abs(el) ≤ 1e-5
            @test logpdf(flow, x) + el ≈ logp(x)
        end

        @testset "likelihood" begin
            sample_trained = rand(flow, 1000)
            sample_untrained = rand(q₀, 1000)
            llh_trained = NormalizingFlows.loglikelihood(rng, flow, sample_trained)
            llh_untrained = NormalizingFlows.loglikelihood(rng, flow, sample_untrained)

            @test llh_trained > llh_untrained
        end
    end
end
