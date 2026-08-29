using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using NormalizingFlows
using ADTypes, Bijectors, CUDA, Distributions, Flux, Functors, LinearAlgebra, Optimisers
using Random, Test
using Mooncake, Zygote
# loads the AbstractPPL extension that routes `AutoZygote` through DifferentiationInterface
import DifferentiationInterface as DI

# keep q0 parameters out of Optimisers.destructure
@leaf MvNormal

# Bijectors' planar layer broadcasts in a way that CUDA cannot fuse, and reads `flow.b` back
# from the device.
# https://github.com/TuringLang/Bijectors.jl/blob/93cb25563043c527905519d81d6dee7917af4dbe/src/bijectors/planar_layer.jl#L65-L110
function Bijectors.get_u_hat(u::CuVector{T}, w::CuVector{T}) where {T<:Real}
    wT_u = dot(w, u)
    scale = (Bijectors.LogExpFunctions.log1pexp(-wT_u) - 1) / sum(abs2, w)
    û = CUDA.broadcast(+, u, CUDA.broadcast(*, scale, w))
    wT_û = Bijectors.LogExpFunctions.log1pexp(wT_u) - 1
    return û, wT_û
end
function Bijectors._transform(flow::PlanarLayer, z::CuArray{T}) where {T<:Real}
    w = CuArray(flow.w)

    û, wT_û = Bijectors.get_u_hat(CuArray(flow.u), w)
    wT_z = Bijectors.aT_b(w, z)

    # `flow.b` holds one element, so broadcasting it avoids reading back from the device.
    tanh_term = CUDA.tanh.(CUDA.broadcast(+, wT_z, flow.b))
    transformed = CUDA.broadcast(+, z, CUDA.broadcast(*, û, tanh_term))

    return (transformed=transformed, wT_û=wT_û, wT_z=wT_z)
end
function Bijectors.with_logabsdet_jacobian(
    flow::PlanarLayer, z::CuMatrix{T}
) where {T<:Real}
    transformed, wT_û, wT_z = Bijectors._transform(flow, z)
    logjac = log1p.(wT_û .* abs2.(sech.(vec(wT_z) .+ flow.b)))
    return (result=transformed, logabsdetjac=logjac)
end

@testset "rand with CUDA" begin
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

# Planar layers throughout, because coupling layers partition through a host sparse
# `PartitionMask`.
@testset "batched ELBO on CUDA" begin
    CUDA.allowscalar(false)
    q0 = MvNormal(CUDA.zeros(Float32, 2), cu(Matrix{Float32}(I, 2, 2)))
    xs = NormalizingFlows._device_specific_rand(CUDA.default_rng(), q0, 64)

    @testset "log-density stays on the device" begin
        lp = NormalizingFlows._device_specific_logpdf(q0, xs)
        @test lp isa CuArray{Float32}
        @test length(lp) == 64
        cpu_q0 = MvNormal(zeros(Float32, 2), Matrix{Float32}(I, 2, 2))
        @test Array(lp) ≈ logpdf(cpu_q0, Array(xs)) rtol = 1.0f-4
    end

    @testset "batched ELBO" begin
        target = MvNormal(CUDA.zeros(Float32, 2), cu(Matrix{Float32}(I, 2, 2)))
        logp(z) = NormalizingFlows._device_specific_logpdf(target, z)
        pl = PlanarLayer(
            CUDA.rand(Float32, 2), CUDA.rand(Float32, 2), CUDA.rand(Float32, 1)
        )
        flow = Bijectors.transformed(q0, pl)

        elbos = NormalizingFlows._batched_elbos(flow, logp, xs)
        @test elbos isa CuArray{Float32}
        @test all(isfinite, Array(elbos))
        @test isfinite(elbo_batch(flow, logp, xs))

        cpu_q0 = MvNormal(zeros(Float32, 2), Matrix{Float32}(I, 2, 2))
        cpu_pl = fmap(Array, pl)
        cpu_flow = Bijectors.transformed(cpu_q0, cpu_pl)
        cpu_target = MvNormal(zeros(Float32, 2), Matrix{Float32}(I, 2, 2))
        cpu_logp(z) = logpdf(cpu_target, z)
        cpu_elbos = NormalizingFlows._batched_elbos(cpu_flow, cpu_logp, Array(xs))
        @test Array(elbos) ≈ cpu_elbos rtol = 1.0f-4

        # Each differentiated use of a full covariance leaves a cotangent of a different
        # matrix type, and summing them indexes the device array.
        g = only(
            Zygote.gradient(x -> sum(NormalizingFlows._batched_elbos(flow, logp, x)), xs)
        )
        @test g isa CuArray{Float32}
        @test all(isfinite, Array(g))
    end
end

@testset "planar flow training on CUDA" begin
    CUDA.allowscalar(false)
    T = Float32
    d = 2

    # Diagonal covariances take the broadcast branch of the log-density. The full covariance
    # branch is covered above.
    q0 = MvNormal(CUDA.zeros(T, d), Diagonal(CUDA.ones(T, d)))
    target = MvNormal(cu(T[2, -1]), Diagonal(CUDA.ones(T, d)))
    logp(z) = NormalizingFlows._device_specific_logpdf(target, z)

    backends = ADTypes.AbstractADType[ADTypes.AutoZygote()]
    # On Julia 1.10 the broadcast kernel reaches Mooncake as a KernelAbstractions foreign
    # call, which it does not differentiate.
    if VERSION >= v"1.11"
        push!(backends, ADTypes.AutoMooncake(; config=Mooncake.Config()))
    end

    @testset "$(nameof(typeof(ad)))" for ad in backends
        layers = [
            PlanarLayer(CUDA.rand(T, d), CUDA.rand(T, d), CUDA.rand(T, 1)) for _ in 1:2
        ]
        flow = create_flow(layers, q0)

        θ, re = Optimisers.destructure(flow)
        @test θ isa CuArray{T}

        flow_trained, stats, _ = train_flow(
            CUDA.default_rng(),
            elbo_batch,
            flow,
            logp,
            32;
            max_iters=5,
            optimiser=Optimisers.Adam(T(1e-3)),
            ADbackend=ad,
            show_progress=false,
        )

        @test all(isfinite, map(x -> x.loss, stats))
        @test Optimisers.destructure(flow_trained)[1] isa CuArray{T}
    end
end
