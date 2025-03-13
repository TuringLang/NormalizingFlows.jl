using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(; path=joinpath(@__DIR__, "..", "..", ".."))

using NormalizingFlows
using CUDA, Distributions, Flux, LinearAlgebra, Test

@testset "rand with CUDA" begin
    dists = [
        MvNormal(CUDA.zeros(2), cu(Matrix{Float64}(I, 2, 2))),
        MvNormal(CUDA.zeros(2), cu([1.0 0.5; 0.5 1.0])),
    ]

    @testset "$dist" for dist in dists
        x = rand_device(CUDA.default_rng(), dist)
        xs = rand_device(CUDA.default_rng(), dist, 100)
        @test x isa CuArray
        @test xs isa CuArray
    end

    @testset "$dist" for dist in dists
        CUDA.allowscalar(true)
        ts = reduce(∘, [Bijectors.PlanarLayer(2) for _ in 1:2])
        ts_g = gpu(ts)
        flow = Bijectors.transformed(dist, ts_g)

        y = rand_device(CUDA.default_rng(), flow)
        ys = rand_device(CUDA.default_rng(), flow, 100)
        @test y isa CuArray
        @test ys isa CuArray
    end
end
