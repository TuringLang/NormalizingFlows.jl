@testset "learining 2d Gaussian" begin
    chuncksize = 4
    @testset "$adtype" for adtype in [
        ADTypes.AutoZygote(),
        ADTypes.AutoForwardDiff(chuncksize),
        ADTypes.AutoForwardDiff(),
        ADTypes.AutoReverseDiff(false),
        # ADTypes.AutoEnzyme(), # doesn't work for Enzyme
    ]
        @testset "$T" for T in [Float32, Float64]
            μ = 10 * ones(T, 2)
            Σ = Diagonal(4 * ones(T, 2))
            target = MvNormal(μ, Σ)
            logp(z) = logpdf(target, z)

            q₀ = MvNormal(zeros(T, 2), ones(T, 2))
            flow = Bijectors.transformed(
                q₀, Bijectors.Shift(zero.(μ)) ∘ Bijectors.Scale(ones(T, 2))
            )

            sample_per_iter = 10
            cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
            checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
            flow_trained, stats, _ = train_flow(
                elbo,
                flow,
                logp,
                sample_per_iter;
                max_iters=5_000,
                optimiser=Optimisers.ADAM(0.01 * one(T)),
                ADbackend=adtype,
                show_progress=false,
                callback=cb,
                hasconverged=checkconv,
            )
            θ, re = Optimisers.destructure(flow_trained)

            el_untrained = elbo(flow, logp, 1000)
            el_trained = elbo(flow_trained, logp, 1000)

            @test all(abs.(θ[1:2] .- μ) .< 0.2)
            @test all(abs.(θ[3:4] .- 2) .< 0.2)
            @test el_trained > el_untrained
            @test el_trained > -1
        end
    end
end