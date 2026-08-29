@testset "variational objectives" begin
    @testset "$T" for T in [Float32, Float64]
        μ = randn(T, 2)
        Σ = Diagonal(rand(T, 2) .+ T(1e-3))
        target = MvNormal(μ, Σ)
        logp(z) = logpdf(target, z)

        q₀ = MvNormal(zeros(T, 2), Diagonal(ones(T, 2)))
        flow = Bijectors.transformed(q₀, Bijectors.Shift(μ) ∘ Bijectors.Scale(sqrt.(Σ)))

        x = randn(T, 2)
        rng = Random.default_rng()

        @testset "elbo" begin
            el = elbo(rng, flow, logp, 10)

            @test abs(el) ≤ 1e-5
            @test logpdf(flow, x) + el ≈ logp(x)
        end

        @testset "elbo_batch" begin
            el = elbo_batch(rng, flow, logp, 10)

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

@testset "batched MvNormal log-density" begin
    @testset "$T" for T in (Float32, Float64)
        # a non-unit scale, so dropping the whitening is caught in this case too
        dists = (
            MvNormal(zeros(T, 3), PDMats.ScalMat(3, T(2))),
            MvNormal(T[1, -2, 0.5], Diagonal(T[2, 0.5, 1])),
            MvNormal(zeros(T, 3), T[2 0.3 0.1; 0.3 1 0.2; 0.1 0.2 1.5]),
        )
        @testset "$(nameof(typeof(d.Σ)))" for d in dists
            xs = randn(T, 3, 6)
            @test NormalizingFlows._cov_logdet(d.Σ) ≈ logdet(d.Σ) rtol = sqrt(eps(T))

            batched = NormalizingFlows._batched_mvnormal_logpdf(d, xs)
            @test eltype(batched) == T
            @test batched ≈ logpdf(d, xs) rtol = sqrt(eps(T))
            @test NormalizingFlows._device_specific_logpdf(d, xs) == logpdf(d, xs)

            loss(v) = sum(NormalizingFlows._batched_mvnormal_logpdf(d, reshape(v, 3, 6)))
            g_ref = ForwardDiff.gradient(loss, vec(xs))
            @test all(isfinite, g_ref)
            @test only(Zygote.gradient(loss, vec(xs))) ≈ g_ref rtol = sqrt(eps(T))
            @test ReverseDiff.gradient(loss, vec(xs)) ≈ g_ref rtol = sqrt(eps(T))
        end
    end
end
