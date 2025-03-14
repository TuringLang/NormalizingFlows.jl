using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using NormalizingFlows
using Bijectors, CUDA, Distributions, Flux, LinearAlgebra, Test

@testset "rand with CUDA" begin

    # Bijectors versions use dot for broadcasting, which causes issues with CUDA.
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

    dists = [
        MvNormal(CUDA.zeros(2), cu(Matrix{Float64}(I, 2, 2))),
        MvNormal(CUDA.zeros(2), cu([1.0 0.5; 0.5 1.0])),
    ]

    @testset "$dist" for dist in dists
        x = NormalizingFlows.rand_device(CUDA.default_rng(), dist)
        xs = NormalizingFlows.rand_device(CUDA.default_rng(), dist, 100)
        @test_nowarn logpdf(dist, x)
        @test x isa CuArray
        @test xs isa CuArray
    end

    @testset "$dist" for dist in dists
        CUDA.allowscalar(true)
        pl1 = PlanarLayer(
            identity(CUDA.rand(2)), identity(CUDA.rand(2)), identity(CUDA.rand(1))
        )
        pl2 = PlanarLayer(
            identity(CUDA.rand(2)), identity(CUDA.rand(2)), identity(CUDA.rand(1))
        )
        flow = Bijectors.transformed(dist, ComposedFunction(pl1, pl2))

        y = NormalizingFlows.rand_device(CUDA.default_rng(), flow)
        ys = NormalizingFlows.rand_device(CUDA.default_rng(), flow, 100)
        @test_nowarn logpdf(flow, y)
        @test y isa CuArray
        @test ys isa CuArray
    end
end
