@testset "RealNVP flow" begin
    Random.seed!(123)
    
    dim = 5
    nlayers = 2
    hdims = [32, 32]
    for T in [Float32, Float64]
        # Create a RealNVP flow
        q₀ = MvNormal(zeros(T, dim), I)
        @leaf MvNormal
        flow = NormalizingFlows.realnvp(q₀, hdims, nlayers; paramtype=T)

        @testset "Sampling and density estimation for type: $T" begin
            ys = rand(flow, 100) 
            ℓs = logpdf(flow, ys)

            @test size(ys) == (dim, 100)
            @test length(ℓs) == 100            

            @test eltype(ys) == T
            @test eltype(ℓs) == T
        end
            

        @testset "Inverse compatibility for type: $T" begin
            x = rand(q₀)
            y, lj_fwd = Bijectors.with_logabsdet_jacobian(flow.transform, x)
            x_reconstructed, lj_bwd = Bijectors.with_logabsdet_jacobian(inverse(flow.transform), y)

            @test x ≈ x_reconstructed rtol=1e-6
            @test lj_fwd ≈ -lj_bwd rtol=1e-6

            x_batch = rand(q₀, 10)
            y_batch, ljs_fwd = Bijectors.with_logabsdet_jacobian(flow.transform, x_batch)
            x_batch_reconstructed, ljs_bwd = Bijectors.with_logabsdet_jacobian(inverse(flow.transform), y_batch)

            @test x_batch ≈ x_batch_reconstructed rtol=1e-6
            @test ljs_fwd ≈ -ljs_bwd rtol=1e-6
        end


        @testset "ELBO test for type: $T" begin
            μ = randn(T, dim)
            Σ = Diagonal(rand(T, dim) .+ T(1e-3))
            target = MvNormal(μ, Σ)
            logp(z) = logpdf(target, z)

            # Define a simple log-likelihood function
            logp(z) = logpdf(q₀, z)

            # Compute ELBO
            batchsize = 64
            elbo_value = elbo(Random.default_rng(), flow, logp, batchsize)
            elbo_batch_value = elbo_batch(Random.default_rng(), flow, logp, batchsize)

            # test elbo_value is not NaN and not Inf
            @test !isnan(elbo_value)
            @test !isinf(elbo_value)
            @test !isnan(elbo_batch_value)
            @test !isinf(elbo_batch_value)
        end

        #todo add tests for ad
    end
end

@testset "Neural Spline flow" begin
    Random.seed!(123)
    
    dim = 5
    nlayers = 2
    hdims = [32, 32]
    for T in [Float32, Float64]
        # Create a RealNVP flow
        q₀ = MvNormal(zeros(T, dim), I)
        @leaf MvNormal
        flow = NormalizingFlows.nsf(q₀; paramtype=T)

        @testset "Sampling and density estimation for type: $T" begin
            ys = rand(flow, 100) 
            ℓs = logpdf(flow, ys)

            @test size(ys) == (dim, 100)
            @test length(ℓs) == 100            

            @test eltype(ys) == T
            @test eltype(ℓs) == T
        end
            

        @testset "Inverse compatibility for type: $T" begin
            x = rand(q₀)
            y, lj_fwd = Bijectors.with_logabsdet_jacobian(flow.transform, x)
            x_reconstructed, lj_bwd = Bijectors.with_logabsdet_jacobian(inverse(flow.transform), y)

            @test x ≈ x_reconstructed rtol=1e-6
            @test lj_fwd ≈ -lj_bwd rtol=1e-6

            x_batch = rand(q₀, 10)
            y_batch, ljs_fwd = Bijectors.with_logabsdet_jacobian(flow.transform, x_batch)
            x_batch_reconstructed, ljs_bwd = Bijectors.with_logabsdet_jacobian(inverse(flow.transform), y_batch)

            @test x_batch ≈ x_batch_reconstructed rtol=1e-6
            @test ljs_fwd ≈ -ljs_bwd rtol=1e-6
        end


        @testset "ELBO test for type: $T" begin
            μ = randn(T, dim)
            Σ = Diagonal(rand(T, dim) .+ T(1e-3))
            target = MvNormal(μ, Σ)
            logp(z) = logpdf(target, z)

            # Define a simple log-likelihood function
            logp(z) = logpdf(q₀, z)

            # Compute ELBO
            batchsize = 64
            elbo_value = elbo(Random.default_rng(), flow, logp, batchsize)
            elbo_batch_value = elbo_batch(Random.default_rng(), flow, logp, batchsize)

            # test elbo_value is not NaN and not Inf
            @test !isnan(elbo_value)
            @test !isinf(elbo_value)
            @test !isnan(elbo_batch_value)
            @test !isinf(elbo_batch_value)
        end

        #todo add tests for ad
    end
end
