@testset "DI.AD with context wrapper" begin
    f(x, y, z) = sum(abs2, x .+ y .+ z)
    T = Float32

    @testset "$T" for T in [Float32, Float64]
        x = randn(T, 10)
        y = randn(T, 10)
        z = randn(T, 10)
        chunksize = size(x, 1)

        @testset "$at" for at in [
            ADTypes.AutoZygote(),
            ADTypes.AutoForwardDiff(; chunksize=chunksize),
            ADTypes.AutoForwardDiff(),
            ADTypes.AutoReverseDiff(; compile=false),
            ADTypes.AutoMooncake(; config=Mooncake.Config()),
        ]
            at = ADTypes.AutoMooncake(; config=Mooncake.Config())
            prep = NormalizingFlows._prepare_gradient(f, at, x, y, z)
            value, grad = NormalizingFlows._value_and_gradient(f, prep, at, x, y, z)
            @test DiffResults.value(out) ≈ f(x, y, z)
            @test DiffResults.gradient(out) ≈ 2 * (x .+ y .+ z)
        end
    end
end

@testset "AD for ELBO" begin
    @testset "$at" for at in [
        ADTypes.AutoZygote(),
        ADTypes.AutoForwardDiff(),
        ADTypes.AutoReverseDiff(; compile = false),
        ADTypes.AutoMooncake(; config=Mooncake.Config()),
    ]
        @testset "$T" for T in [Float32, Float64]
            μ = 10 * ones(T, 2)
            Σ = Diagonal(4 * ones(T, 2))
            target = MvNormal(μ, Σ)
            logp(z) = logpdf(target, z)
            
            # necessary for Zygote/mooncake to differentiate through the flow
            # prevent opt q0
            @leaf MvNormal 
            q₀ = MvNormal(zeros(T, 2), ones(T, 2))
            flow = Bijectors.transformed(q₀, Bijectors.Shift(zero.(μ)))

            sample_per_iter = 10
            θ, re = Optimisers.destructure(flow)

            # check grad computation for elbo
            loss(θ, args...) = -NormalizingFlows.elbo(re(θ), args...)
            prep = NormalizingFlows._prepare_gradient(loss, at, θ, logp, randn(T, 2, sample_per_iter))
            value, grad = NormalizingFlows._value_and_gradient(
                loss, prep, at, θ, logp, randn(T, 2, sample_per_iter)
            )

            @test !isnothing(value)
            @test all(grad .!= nothing)
        end
    end
end
