# Training a planar flow on the GPU. This demo has its own project so that the CPU examples
# do not pull in CUDA. Run it from `example/gpu` after
#
#     using Pkg; Pkg.activate("."); Pkg.develop(; path="../.."); Pkg.instantiate()
#
# The `develop` is needed until 0.4.1 is registered, because the device log-density this
# demo relies on landed in that version.
#
# Coupling flows (RealNVP, NSF) do not run on the GPU yet: `Bijectors.PartitionMask` holds
# host sparse matrices and `partition`/`combine` multiply against them.

using Distributions, LinearAlgebra
using Bijectors
using Functors
using Optimisers, ADTypes, Zygote
using CUDA
using NormalizingFlows
# loads the AbstractPPL extension that routes `AutoZygote` through DifferentiationInterface
using DifferentiationInterface

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
function Bijectors._transform(flow::Bijectors.PlanarLayer, z::CuArray{T}) where {T<:Real}
    w = CuArray(flow.w)
    û, wT_û = Bijectors.get_u_hat(CuArray(flow.u), w)
    wT_z = Bijectors.aT_b(w, z)
    # `flow.b` holds one element, so broadcasting it is the same as Bijectors' `first(flow.b)`
    # without reading back from the device.
    tanh_term = CUDA.tanh.(CUDA.broadcast(+, wT_z, flow.b))
    transformed = CUDA.broadcast(+, z, CUDA.broadcast(*, û, tanh_term))
    return (transformed=transformed, wT_û=wT_û, wT_z=wT_z)
end
function Bijectors.with_logabsdet_jacobian(
    flow::Bijectors.PlanarLayer, z::CuMatrix{T}
) where {T<:Real}
    transformed, wT_û, wT_z = Bijectors._transform(flow, z)
    logjac = log1p.(wT_û .* abs2.(sech.(vec(wT_z) .+ flow.b)))
    return (result=transformed, logabsdetjac=logjac)
end

rng = CUDA.default_rng()
T = Float32
d = 2

@leaf MvNormal
q0 = MvNormal(CUDA.zeros(T, d), Diagonal(CUDA.ones(T, d)))

# `logp` takes the whole `(d, n)` batch and returns one value per column. Writing it with
# array operations keeps it on the device, where `logpdf` would gather the columns onto the
# host. The normaliser is kept so the reported ELBO is the true one.
const μ_target = cu(T[2, -1])
const logZ = T(d * log(2 * π))
logp(z) = vec(-(logZ .+ sum(abs2, z .- μ_target; dims=1)) ./ 2)

layers = [
    Bijectors.PlanarLayer(CUDA.rand(T, d), CUDA.rand(T, d), CUDA.rand(T, 1)) for _ in 1:4
]
flow = create_flow(layers, q0)

sample_per_iter = 64
flow_trained, stats, _ = train_flow(
    rng,
    elbo_batch,
    flow,
    logp,
    sample_per_iter;
    max_iters=2_000,
    optimiser=Optimisers.Adam(one(T) / 100),
    # Zygote rather than Mooncake: Mooncake 0.5.48 fails on this flow inside its own CUDA
    # kernel launch, a `CoDual` type assertion in `Adapt.adapt_storage`.
    ADbackend=ADTypes.AutoZygote(),
)

losses = map(x -> x.loss, stats)
@info "ELBO" start = -losses[1] final = -losses[end]

ys = NormalizingFlows._device_specific_rand(rng, flow_trained, 1_000)
@info "posterior mean (target is $(Array(μ_target)))" mean(Array(ys); dims=2)
