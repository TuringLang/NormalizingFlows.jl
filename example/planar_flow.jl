include("../src/NormalizingFlows.jl")
using .NormalizingFlows
using Random, Distributions, LinearAlgebra, Bijectors
using Plots
using Optimisers
import AbstractDifferentiation as AD

Random.seed!(123)
rng = Random.default_rng()

######################################
# Define banana distribution as target
######################################
struct Banana{T<:Real} <: ContinuousMultivariateDistribution
    d::Int            # Dimension
    b::T        # Curvature
    Z::T        # Normalizing constant
    C::Matrix{T} # Covariance matrix
    C⁻¹::Matrix{T} # Inverse of covariance matrix
end

# Constructor with additional scaling parameter s
function Banana(d::Int, b::T, s::T=100.0f0) where {T<:Real}
    return Banana(
        d,
        b,
        T(sqrt(s * (2π)^d)),
        Matrix(Diagonal(vcat(s, ones(T, d - 1)))),
        Matrix(Diagonal(vcat(1 / s, ones(T, d - 1)))),
    )
end
Base.length(p::Banana) = p.d

Distributions.sampler(p::Banana) = p

# Define the transformation function φ and the inverse ϕ⁻¹ for the banana distribution
φ(x, b, s) = [x[1], x[2] + b * x[1]^2 - s * b]
ϕ⁻¹(y, b, s) = [y[1], y[2] - b * y[1]^2 + s * b]

function Distributions._rand!(rng::AbstractRNG, p::Banana, x::AbstractArray{<:Real})
    b, C = p.b, p.C
    mvnormal = MvNormal(zeros(2), C)
    for i in axes(x, 2)
        x[:, i] = φ(rand(rng, mvnormal), b, C[1, 1])
    end
    return x
end

function Distributions._logpdf(p::Banana, x::AbstractArray)
    Z, C⁻¹, b = p.Z, p.C⁻¹, p.b
    ϕ⁻¹_x = ϕ⁻¹(x, b, p.C[1, 1])
    return -log(Z) - ϕ⁻¹_x' * C⁻¹ * ϕ⁻¹_x / 2
end

function visualize(p::Banana, samples=rand(p, 1000))
    xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1; length=100)
    yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1; length=100)
    z = [exp(Distributions.logpdf(p, [x, y])) for x in xrange, y in yrange]
    contour(xrange, yrange, z'; levels=15, color=:viridis, label="PDF", linewidth=2)
    scatter!(samples[1, :], samples[2, :]; label="Samples", alpha=0.3, legend=:bottomright)
    return current()
end

######################################
# learn the target using planar flow 
######################################
function create_planar_flow(n_layers::Int, q₀)
    d = length(q₀)
    Ts = ∘([PlanarLayer(d) for _ in 1:n_layers]...)
    return transformed(q₀, Ts)
end

# create target p
p = Banana(2, 1.0f-1)
logp = Base.Fix1(logpdf, p)
# visualize(p)

# create a 10-layer planar flow
flow = create_planar_flow(15, MvNormal(zeros(Float32, 2), I))
flow_untrained = deepcopy(flow)

# train the flow
sample_per_iter = 1
flow_trained, losses, _ = NF(
    elbo,
    flow,
    logp,
    sample_per_iter;
    rng=rng,
    max_iters=100000,
    optimiser=Optimisers.ADAM(),
    AD_backend=AD.ZygoteBackend(),
)

######################################
# evaluate the trained flow
######################################

function compare_trained_and_untrained_flow(
    flow_trained, flow_untrained, true_dist, n_samples
)
    samples_trained = rand(flow_trained, n_samples)
    samples_untrained = rand(flow_untrained, n_samples)
    samples_true = rand(true_dist, n_samples)

    scatter(
        samples_true[1, :],
        samples_true[2, :];
        label="True Distribution",
        color=:blue,
        markersize=2,
        alpha=0.5,
    )
    scatter!(
        samples_untrained[1, :],
        samples_untrained[2, :];
        label="Untrained Flow",
        color=:red,
        markersize=2,
        alpha=0.5,
    )
    scatter!(
        samples_trained[1, :],
        samples_trained[2, :];
        label="Trained Flow",
        color=:green,
        markersize=2,
        alpha=0.5,
    )

    xlabel!("X")
    ylabel!("Y")
    title!("Comparison of Trained and Untrained Flow")

    return current()
end

plot(losses; label="Loss", linewidth=2)
compare_trained_and_untrained_flow(flow_trained, flow_untrained, p, 1000)
