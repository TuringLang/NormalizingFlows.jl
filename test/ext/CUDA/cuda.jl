using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using NormalizingFlows
using Bijectors, CUDA, Distributions, Flux, LinearAlgebra, Random, Test
using Zygote

@testset "rand with CUDA" begin

    # Bijectors versions use dot for broadcasting, which causes issues with CUDA.
    # https://github.com/TuringLang/Bijectors.jl/blob/6f0d383f73afd150a018b65a3ea4ac9306065d38/src/bijectors/planar_layer.jl#L65-L80
    function Bijectors.get_u_hat(u::CuVector{T}, w::CuVector{T}) where {T<:Real}
        wT_u = dot(w, u)
        scale = (Bijectors.LogExpFunctions.log1pexp(-wT_u) - 1) / sum(abs2, w)
        û = CUDA.broadcast(+, u, CUDA.broadcast(*, scale, w))
        wT_û = Bijectors.LogExpFunctions.log1pexp(wT_u) - 1
        return û, wT_û
    end
    function Bijectors._transform(flow::PlanarLayer, z::CuArray{T}) where {T<:Real}
        w = CuArray(flow.w)
        b = T(first(flow.b))  # Scalar

        û, wT_û = Bijectors.get_u_hat(CuArray(flow.u), w)
        wT_z = Bijectors.aT_b(w, z)

        tanh_term = CUDA.tanh.(CUDA.broadcast(+, wT_z, b))
        transformed = CUDA.broadcast(+, z, CUDA.broadcast(*, û, tanh_term))

        return (transformed=transformed, wT_û=wT_û, wT_z=wT_z)
    end

    CUDA.allowscalar(true)
    dists = [
        MvNormal(CUDA.zeros(2), cu(Matrix{Float64}(I, 2, 2))),
        MvNormal(CUDA.zeros(2), cu([1.0 0.5; 0.5 1.0])),
    ]

    @testset "$dist" for dist in dists
        x = NormalizingFlows._device_specific_rand(CUDA.default_rng(), dist)
        xs = NormalizingFlows._device_specific_rand(CUDA.default_rng(), dist, 100)
        @test_nowarn logpdf(dist, x)
        @test x isa CuArray
        @test xs isa CuArray
    end

    @testset "$dist" for dist in dists
        pl1 = PlanarLayer(
            identity(CUDA.rand(2)), identity(CUDA.rand(2)), identity(CUDA.rand(1))
        )
        pl2 = PlanarLayer(
            identity(CUDA.rand(2)), identity(CUDA.rand(2)), identity(CUDA.rand(1))
        )
        flow = Bijectors.transformed(dist, ComposedFunction(pl1, pl2))

        y = NormalizingFlows._device_specific_rand(CUDA.default_rng(), flow)
        ys = NormalizingFlows._device_specific_rand(CUDA.default_rng(), flow, 100)
        @test_nowarn logpdf(flow, y)
        @test y isa CuArray
        @test ys isa CuArray
    end
end

@testset "RQS on CUDA" begin
    CUDA.allowscalar(false)

    rng = MersenneTwister(1)
    K, D, N = 4, 3, 8
    B = 5.0f0
    θ_cpu = randn(rng, Float32, (3K - 1) * D, N)
    x_cpu = 3.0f0 .* randn(rng, Float32, D, N)
    θ_gpu = cu(θ_cpu)
    x_gpu = cu(x_cpu)

    params_cpu = NormalizingFlows.rqs_params_from_raw(θ_cpu, D, B)
    params_gpu = NormalizingFlows.rqs_params_from_raw(θ_gpu, D, B)
    @test all(p -> p isa CuArray{Float32}, params_gpu)
    for (p_cpu, p_gpu) in zip(params_cpu, params_gpu)
        @test Array(p_gpu) ≈ p_cpu rtol = 1.0f-4
    end

    @testset "$f" for f in (NormalizingFlows.rqs_forward, NormalizingFlows.rqs_inverse)
        out_cpu, logjac_cpu = f(x_cpu, params_cpu...)
        out_gpu, logjac_gpu = f(x_gpu, params_gpu...)
        @test out_gpu isa CuArray{Float32}
        @test logjac_gpu isa CuArray{Float32}
        @test Array(out_gpu) ≈ out_cpu rtol = 1.0f-4
        @test Array(logjac_gpu) ≈ logjac_cpu rtol = 1.0f-4

        loss(θ, x) = sum(last(f(x, NormalizingFlows.rqs_params_from_raw(θ, D, B)...)))
        g_cpu = only(Zygote.gradient(θ -> loss(θ, x_cpu), θ_cpu))
        g_gpu = only(Zygote.gradient(θ -> loss(θ, x_gpu), θ_gpu))
        @test g_gpu isa CuArray{Float32}
        @test Array(g_gpu) ≈ g_cpu rtol = 1.0f-3
    end

    y_gpu, _ = NormalizingFlows.rqs_forward(x_gpu, params_gpu...)
    x_back, _ = NormalizingFlows.rqs_inverse(y_gpu, params_gpu...)
    @test Array(x_back) ≈ x_cpu rtol = 1.0f-4
end
