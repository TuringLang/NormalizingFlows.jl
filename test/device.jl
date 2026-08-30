# JLArrays runs on the CPU but rejects scalar indexing, so it reaches device-only failures
# without a GPU.

# cuSOLVER returns `uplo = 'U'`, which is what makes PDMats wrap the factors in an `Adjoint`.
function device_pdmat(A::AbstractMatrix)
    c = cholesky(A)
    return PDMats.PDMat(jl(A), Cholesky(jl(Matrix(c.factors)), c.uplo, c.info))
end

@testset "batched MvNormal log-density on device arrays" begin
    GPUArraysCore.allowscalar(false)

    @testset "$T" for T in (Float32, Float64)
        rtol = T == Float32 ? 1.0f-4 : 1.0e-8
        xs = randn(T, 2, 5)
        covariances = (
            PDMats.ScalMat(2, one(T)),
            PDMats.PDiagMat(T[2, 0.5]),
            PDMats.PDMat(T[2 0.3; 0.3 1]),
        )

        @testset "$(nameof(typeof(Σ)))" for Σ in covariances
            host = MvNormal(zeros(T, 2), Σ)
            Σ_dev = if Σ isa PDMats.PDMat
                device_pdmat(Matrix(Σ))
            elseif Σ isa PDMats.PDiagMat
                PDMats.PDiagMat(jl(Σ.diag))
            else
                Σ
            end
            dev = MvNormal(jl(zeros(T, 2)), Σ_dev)
            xs_dev = jl(xs)

            batched = NormalizingFlows._batched_mvnormal_logpdf(dev, xs_dev)
            @test batched isa JLArray
            @test Array(batched) ≈ logpdf(host, xs) rtol = rtol

            @test NormalizingFlows._device_specific_logpdf(dev, xs_dev) ≈ batched rtol =
                rtol
            @test NormalizingFlows._device_specific_logpdf(host, xs) == logpdf(host, xs)

            # A full covariance leaves one cotangent per differentiated use, and summing them
            # indexes the device array.
            g = only(
                Zygote.gradient(
                    x -> sum(NormalizingFlows._batched_mvnormal_logpdf(dev, x)), xs_dev
                ),
            )
            @test g isa JLArray
            g_ref = ForwardDiff.gradient(x -> sum(logpdf(host, reshape(x, 2, 5))), vec(xs))
            @test Array(g) ≈ reshape(g_ref, 2, 5) rtol = rtol
        end
    end
end
