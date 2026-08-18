using NormalizingFlows: rqs_params_from_raw, rqs_forward, rqs_inverse

@testset "RQS parameters" begin
    @testset "T=$T, K=$K, D=$D, N=$N, B=$B" for T in (Float32, Float64),
        K in (4, 8),
        D in (1, 3),
        N in (1, 16),
        B in (2, 30)

        θ_raw = randn(T, (3K - 1) * D, N)
        widths, heights, derivatives = rqs_params_from_raw(θ_raw, D, B)

        @test size(widths) == (K + 1, D, N)
        @test size(heights) == (K + 1, D, N)
        @test size(derivatives) == (K + 1, D, N)

        @test eltype(widths) == T
        @test eltype(heights) == T
        @test eltype(derivatives) == T

        for grid in (widths, heights)
            @test all(grid[1, :, :] .== -T(B))
            @test all(grid[end, :, :] .== T(B))
            @test all(diff(grid; dims=1) .> 0)
        end

        @test all(derivatives[1, :, :] .== one(T))
        @test all(derivatives[end, :, :] .== one(T))
        @test all(derivatives .> 0)
    end

    @testset "raw parameter layout" begin
        K, D, N, B = 4, 2, 3, 5.0
        base = zeros((3K - 1) * D, N)
        w0, h0, d0 = rqs_params_from_raw(base, D, B)
        for (idx, moved) in ((1, :widths), (K + 1, :heights), (2K + 1, :derivatives))
            θ = copy(base)
            θ[idx, 1] += 1.0
            w, h, d = rqs_params_from_raw(θ, D, B)
            @test (w != w0) == (moved == :widths)
            @test (h != h0) == (moved == :heights)
            @test (d != d0) == (moved == :derivatives)
        end
    end

    @testset "extreme logits stay finite, T=$T" for T in (Float32, Float64)
        # These logits underflow a width, a height, and a slope to zero without the floors.
        K, D, N, B = 4, 1, 1, 5
        θ = zeros(T, 3K - 1, N)
        θ[1, 1] = 110
        θ[K + 1, 1] = 110
        θ[2K + 1, 1] = -150
        w, h, d = rqs_params_from_raw(θ, D, B)

        @test all(diff(w; dims=1) .>= 2 * T(B) * T(1e-3) * T(0.99))
        @test all(diff(h; dims=1) .>= 2 * T(B) * T(1e-3) * T(0.99))
        @test all(d .>= T(1e-3))

        xs = reshape(vcat(T[-0.99B, 0, 0.99B, -B - 1, B + 1], w[:, 1, 1]), 1, :)
        wN = repeat(w; outer=(1, 1, size(xs, 2)))
        hN = repeat(h; outer=(1, 1, size(xs, 2)))
        dN = repeat(d; outer=(1, 1, size(xs, 2)))

        y, logjac = rqs_forward(xs, wN, hN, dN)
        @test all(isfinite, y)
        @test all(isfinite, logjac)
        xb, logjac_inv = rqs_inverse(y, wN, hN, dN)
        @test all(isfinite, xb)
        @test all(isfinite, logjac_inv)

        gx = ForwardDiff.gradient(
            v -> sum(rqs_forward(reshape(v, 1, :), wN, hN, dN)[1]), vec(xs)
        )
        @test all(isfinite, gx)
        gθ = ForwardDiff.gradient(vec(θ)) do v
            wg, hg, dg = rqs_params_from_raw(reshape(v, :, N), D, B)
            sum(rqs_forward(reshape(T[0.99B], 1, 1), wg, hg, dg)[2])
        end
        @test all(isfinite, gθ)
    end
end

@testset "RQS forward" begin
    @testset "T=$T, K=$K, D=$D, N=$N" for T in (Float32, Float64),
        K in (4, 8),
        D in (1, 3),
        N in (1, 8)

        B = 5
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        x = T(0.8B) .* (2 .* rand(T, D, N) .- 1)
        y, logjac = rqs_forward(x, w, h, d)

        @test size(y) == (D, N)
        @test size(logjac) == (1, N)
        @test eltype(y) == T
        @test eltype(logjac) == T

        # Each column reproduces the single-spline evaluation in Bijectors.
        for n in 1:N, i in 1:D
            @test y[i, n] ≈
                Bijectors.rqs_univariate(w[:, i, n], h[:, i, n], d[:, i, n], x[i, n])
        end

        # The coupling is diagonal per dimension, so exp(logjac[n]) equals the product over
        # dims of dyᵢ/dxᵢ from ForwardDiff.
        for n in 1:N
            prod_dydx = one(T)
            for i in 1:D
                scalar_forward = function (xi)
                    xcol = reshape([j == i ? xi : x[j, n] for j in 1:D], D, 1)
                    return rqs_forward(xcol, w[:, :, n:n], h[:, :, n:n], d[:, :, n:n])[1][i]
                end
                dydx = ForwardDiff.derivative(scalar_forward, x[i, n])
                @test dydx > 0
                prod_dydx *= dydx
            end
            @test prod_dydx ≈ exp(logjac[1, n])
        end
    end

    @testset "out-of-range identity, T=$T" for T in (Float32, Float64)
        B = 5
        K, D, N = 6, 2, 4
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        x = T[2B -2B 3B -3B; 2B -2B 3B -3B]
        y, logjac = rqs_forward(x, w, h, d)
        @test y == x
        @test all(iszero, logjac)
    end
end

@testset "RQS inverse" begin
    @testset "T=$T, K=$K, D=$D, N=$N" for T in (Float32, Float64),
        K in (4, 8),
        D in (1, 3),
        N in (1, 8)

        B = 5
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        rtol = T == Float32 ? 1.0f-4 : 1.0e-9

        x = T(0.8B) .* (2 .* rand(T, D, N) .- 1)
        y, logjac_fwd = rqs_forward(x, w, h, d)
        xback, logjac_inv = rqs_inverse(y, w, h, d)
        @test xback ≈ x rtol = rtol
        @test logjac_inv ≈ -logjac_fwd rtol = rtol

        yin = T(0.8B) .* (2 .* rand(T, D, N) .- 1)
        xr, _ = rqs_inverse(yin, w, h, d)
        yr, _ = rqs_forward(xr, w, h, d)
        @test yr ≈ yin rtol = rtol
    end

    @testset "out-of-range identity, T=$T" for T in (Float32, Float64)
        B = 5
        K, D, N = 6, 2, 4
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        y = T[2B -2B 3B -3B; 2B -2B 3B -3B]
        x, logjac = rqs_inverse(y, w, h, d)
        @test x == y
        @test all(iszero, logjac)
    end

    @testset "boundary gradient is finite, T=$T" for T in (Float32, Float64)
        B = 5
        K, D, N = 6, 1, 5
        w, h, d = rqs_params_from_raw(randn(T, (3K - 1) * D, N), D, B)
        y = reshape(T[-B - 1, -B, 0, B, B + 1], D, N)
        for part in (1, 2)
            g = ForwardDiff.gradient(
                v -> sum(rqs_inverse(reshape(v, D, N), w, h, d)[part]), vec(y)
            )
            @test all(isfinite, g)
        end
    end
end

@testset "RQS gradients match ForwardDiff" begin
    K, D, N, B = 4, 2, 3, 5
    n_raw = (3K - 1) * D
    rng = Random.MersenneTwister(11)

    # Inputs spanning the spline and both tails, and logits extreme enough to saturate the
    # floors, so every branch is differentiated.
    inputs = Dict(
        "interior" => 0.8B .* randn(rng, D * N),
        "tails" => vcat(0.5B .* randn(rng, 3), [2B, -2B, B + 0.5]),
    )
    raws = Dict(
        "regular" => randn(rng, n_raw * N), "extreme" => copy(randn(rng, n_raw * N))
    )
    raws["extreme"][1] = 110
    raws["extreme"][K + 1] = 110
    raws["extreme"][2K + 1] = -150

    @testset "$f, $rawname raw, $xname inputs" for f in (rqs_forward, rqs_inverse),
        (rawname, raw) in raws,
        (xname, x) in inputs

        θ = vcat(raw, x)
        function loss(θ)
            w, h, d = rqs_params_from_raw(reshape(θ[1:(n_raw * N)], n_raw, N), D, B)
            xs = reshape(θ[(n_raw * N + 1):end], D, N)
            out, logjac = f(xs, w, h, d)
            return sum(out) + sum(logjac)
        end

        g_ref = ForwardDiff.gradient(loss, θ)
        @test all(isfinite, g_ref)
        @test only(Zygote.gradient(loss, θ)) ≈ g_ref rtol = 1e-8
        @test ReverseDiff.gradient(loss, θ) ≈ g_ref rtol = 1e-8
        # Enzyme fails LLVM verification differentiating the spline on Julia 1.10.
        if VERSION >= v"1.11"
            enzyme = AutoEnzyme(;
                mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
                function_annotation=Enzyme.Const,
            )
            @test DI.gradient(loss, enzyme, θ) ≈ g_ref rtol = 1e-8
        end
        @test DI.gradient(loss, AutoMooncake(; config=Mooncake.Config()), θ) ≈ g_ref rtol =
            1e-8
        @test DI.gradient(loss, AutoMooncakeForward(; config=Mooncake.Config()), θ) ≈ g_ref rtol =
            1e-8
    end

    # Gradients with respect to the inputs only, with the knot parameters as plain
    # constants, is the score of a fixed trained flow.
    @testset "input-only gradient, $f" for f in (rqs_forward, rqs_inverse)
        w, h, d = rqs_params_from_raw(reshape(raws["regular"], n_raw, N), D, B)
        x = inputs["tails"]
        function loss(v)
            out, logjac = f(reshape(v, D, N), w, h, d)
            return sum(out) + sum(logjac)
        end
        g_ref = ForwardDiff.gradient(loss, x)
        @test all(isfinite, g_ref)
        @test ReverseDiff.gradient(loss, x) ≈ g_ref rtol = 1e-8
        @test only(Zygote.gradient(loss, x)) ≈ g_ref rtol = 1e-8
    end
end
