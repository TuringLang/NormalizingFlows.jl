@testset "AD correctness" begin
    f(x) = sum(abs2, x)

    @testset "$T" for T in [Float32, Float64]
        x = randn(T, 10)
        chuncksize = size(x, 1)

        @testset "$at" for at in [
            ADTypes.AutoZygote(),
            ADTypes.AutoForwardDiff(chuncksize),
            ADTypes.AutoForwardDiff(),
            ADTypes.AutoReverseDiff(false),
            ADTypes.AutoEnzyme(),
        ]
            out = DiffResults.GradientResult(x)
            NormalizingFlows.value_and_gradient!(at, f, x, out)
            @test DiffResults.value(out) ≈ f(x)
            @test DiffResults.gradient(out) ≈ 2x
        end
    end
end

@testset "AD for ELBO" begin
    @testset "$at" for at in [
        ADTypes.AutoZygote(),
        ADTypes.AutoForwardDiff(),
        ADTypes.AutoReverseDiff(false),
        # ADTypes.AutoEnzyme(), # not working now
    ]
        @testset "$T" for T in [Float32, Float64]
            μ = 10 * ones(T, 2)
            Σ = Diagonal(D .^ 2)
            target = MvNormal(μ, Σ)
            logp(z) = logpdf(target, z)

            q₀ = MvNormal(zeros(T, 2), ones(T, 2))
            flow = Bijectors.transformed(q₀, Bijectors.Shift(zero.(μ)))

            sample_per_iter = 10
            θ, re = Optimisers.destructure(flow)
            out = DiffResults.GradientResult(θ)

            # check grad computation for elbo
            NormalizingFlows.grad!(
                Random.default_rng(), at, elbo, θ, re, out, logp, sample_per_iter
            )

            @test DiffResults.value(out) != nothing
            @test all(DiffResults.gradient(out) .!= nothing)
        end
    end
end