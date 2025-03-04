@testset "DI.AD with context wrapper" begin
    f(x, y, z) = sum(abs2, x .+ y .+ z)

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
            ADTypes.AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse)),
            ADTypes.AutoMooncake(; config=Mooncake.Config()),
        ]
            prep = NormalizingFlows._prepare_gradient(f, at, x, y, z)
            value, grad = NormalizingFlows._value_and_gradient(f, prep, at, x, y, z)
            @test value ≈ f(x, y, z)
            @test grad ≈ 2 * (x .+ y .+ z)
        end
    end
end

@testset "AD for ELBO on mean-field Gaussian VI" begin
    @testset "$at" for at in [
        ADTypes.AutoZygote(),
        ADTypes.AutoForwardDiff(),
        ADTypes.AutoReverseDiff(; compile = false),
        ADTypes.AutoEnzyme(mode=Enzyme.set_runtime_activity(Enzyme.Reverse)),
        # ADTypes.AutoMooncake(; config=Mooncake.Config()),
    ]
        @testset "$T" for T in [Float32, Float64]
            μ = 10 * ones(T, 2)
            Σ = Diagonal(4 * ones(T, 2))
            target = MvNormal(μ, Σ)
            logp(z) = logpdf(target, z)
            
            # necessary for Zygote/mooncake to differentiate through the flow
            # prevent updating params of q0
            @leaf MvNormal 
            q₀ = MvNormal(zeros(T, 2), ones(T, 2))
            flow = Bijectors.transformed(
                q₀, Bijectors.Shift(zeros(T, 2)) ∘ Bijectors.Scale(ones(T, 2))
            )
    
            θ, re = Optimisers.destructure(flow)

            # check grad computation for elbo
            loss(θ, rng, logp, sample_per_iter) = -NormalizingFlows.elbo(rng, re(θ), logp, sample_per_iter)

            rng = Random.default_rng()
            sample_per_iter = 10

            prep = NormalizingFlows._prepare_gradient(loss, at, θ, rng, logp, sample_per_iter)
            value, grad = NormalizingFlows._value_and_gradient(
                loss, prep, at, θ, rng, logp, sample_per_iter
            )

            @test value !== nothing
            @test all(grad .!= nothing)
        end
    end
end
