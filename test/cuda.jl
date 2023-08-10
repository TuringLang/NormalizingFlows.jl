using CUDA, Test, LinearAlgebra, Distributions

if CUDA.functional()
    @testset "rand with CUDA" begin
        dists = [
            MvNormal(CUDA.zeros(2), I),
            MvNormal(CUDA.zeros(2), cu([1.0 0.5; 0.5 1.0])),
            MvLogNormal(CUDA.zeros(2), I),
            MvLogNormal(CUDA.zeros(2), cu([1.0 0.5; 0.5 1.0])),
        ]

        @testset "$dist" for dist in dists
            x = rand(CUDA.default_rng(), dist)
            @info logpdf(dist, x)
            @test x isa CuArray
        end
    end
end
