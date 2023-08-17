using CSV, DataFrames
using Random, Distributions, LinearAlgebra, Bijectors
using ADTypes
using Optimisers
using Tullio: @tullio
using FunctionChains
using NormalizingFlows
using Zygote
using Zygote: @adjoint, Buffer
using Bijectors: Shift, Scale
include("../common.jl")
include("hamiltonian_layer.jl")

##############################33
# model for posterior inference (logistic regression) 
#################################

#########################################################################################
# Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
# The observed data D = {X, y} consist of N binary class labels, 
# y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
# The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
# and a precision parameter \alpha \in R_+. We assume the following model:
#     p(α) = Gamma(α ; a, b) , τ = log α ∈ R  
#     p(w_k | τ) = N(w_k; 0, exp(-τ))
#     p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t)), y ∈ {1, 0}
#########################################################################################
df = DataFrame(CSV.File("example/HamiltonianVI/data/bank_dat.csv"))
xs = Matrix(df)[:, 2:end]
X_raw = xs[:, 1:(end - 1)]
const X = (X_raw .- mean(X_raw; dims=1)) ./ std(X_raw; dims=1)
const Y = xs[:, end]
const a, b = 1.0, 0.01
(N, p) = size(X)
idx = sample(1:N, 20; replace=false)
const Xc, Yc = X[idx, :], Y[idx]

function log_sigmoid(x)
    if x < -300
        return x
    else
        return -log1p(exp(-x))
    end
end

function neg_sigmoid(x)
    return -1.0 / (1.0 + exp(-x))
end

# z = (τ, w1, ..., wd)
function logp(θ, X, Y, w)
    τ = θ[1]
    W = @view(θ[2:end])
    Z = X * W
    logpτ = a * τ - b * exp(τ)
    logpW = 0.5 * p * τ - 0.5 * exp(τ) * sum(abs2, W)
    @tullio llh := w[n] * ((Y[n] - 1.0) * Z[n] + log_sigmoid(Z[n]))
    # llh = sum((Y .- 1.) .* Z .- log1p.(exp.(-Z)))
    return logpτ + logpW + llh
end

function logp_subsample(θ; batch_size=10)
    (N, p) = size(X)
    idx = sample(1:N, batch_size; replace=false)
    w = N / batch_size .* ones(batch_size)
    return logp(θ, X[idx, :], Y[idx], w)
end

function ∇logp(z, X, Y, w)
    τ = z[1]
    W = @view(z[2:end])
    grad = similar(z)
    grad[1] = a - b * exp(τ) + 0.5 * p - 0.5 * exp(τ) * sum(abs2, W)
    S = neg_sigmoid.(X * W)
    @tullio M[j] := w[n] * X[n, j] * (S[n] + Y[n])
    grad[2:end] .= -exp(τ) .* W .+ M
    return grad
end

function ∇logp_subsample(z; batch_size=10)
    (N, p) = size(X)
    idx = sample(1:N, batch_size; replace=false)
    w = N / batch_size .* ones(batch_size)
    return ∇logp(z, X[idx, :], Y[idx], w)
end

function ∇logp_coreset(z, w)
    @assert length(w) == size(Xc, 1)
    τ = z[1]
    W = z[2:end]
    dim = length(z)
    grad = Buffer(z)
    grad[1] = a - b * exp(τ) + 0.5 * p - 0.5 * exp(τ) * sum(abs2, W)
    S = neg_sigmoid.(Xc * W)
    @tullio M[j] := w[n] * Xc[n, j] * (S[n] + Yc[n])
    grad[2:dim] = -exp(τ) .* W .+ M
    return copy(grad)
end

# customize gradient for logp_subsample (in each iteration logp evaluation only uses a small batch of the full dataset)
Zygote.refresh()
@adjoint function logp_subsample(z; batch_size=10)
    return logp_subsample(z; batch_size=batch_size),
    Δ -> (Δ * ∇logp_subsample(z; batch_size=batch_size),)
end

function logp_joint(z; batch_size=10)
    dim = div(length(z), 2)
    x, ρ = z[1:dim], z[(dim + 1):end]
    return logp_subsample(x; batch_size=10) + logpdf(MvNormal(zeros(dims), I), ρ)
end

function ∇logp_joint(z; batch_size=10)
    dim = div(length(z), 2)
    gx = ∇logp_subsample(z[1:dim]; batch_size=batch_size)
    gρ = -z[(dim + 1):end]
    return vcat(gx, gρ)
end

@adjoint function logp_joint(z; batch_size=10)
    return logp_joint(z; batch_size=batch_size),
    Δ -> (Δ * ∇logp_joint(z; batch_size=batch_size),)
end

#################################################33
# train sparse Hamiltonian flow (https://arxiv.org/pdf/2203.05723.pdf) 
# note: 
# - SHF operates on a joint space (target space × momentum space)
# - instead of using the full score in the Hamiltonain flow, we use coreset score 
# - instead of using the momentum refreshment as used in the paper (only perform normalization on the mometnum), 
# we just stack a general shift and scaling layer after the leapfrog step
###################################################3

∇S = CoresetScore(Float64, 20, 400, ∇logp_coreset)
dims = 9
L = 20
∇logm(x) = -x
Ls = [
    Scale(ones(2dims)) ∘ Shift(ones(2dims)) ∘ SurrogateLeapFrog(dims, 0.02, L, ∇S, ∇logm)
    for i in 1:5
]
q0 = MvNormal(zeros(Float64, 2dims), I)
flow = Bijectors.transformed(q0, ∘(Ls...))
# flow = Bijectors.transformed(q0, trans)
flow_untrained = deepcopy(flow)

sample_per_iter = 5
cb(iter, opt_stats, re, θ) = (sample_per_iter=sample_per_iter,)
checkconv(iter, stat, re, θ, st) = stat.gradient_norm < 1e-3
flow_trained, stats, _ = train_flow(
    elbo,
    flow,
    logp_joint,
    sample_per_iter;
    max_iters=200_00,
    optimiser=Optimisers.Adam(1e-3),
    callback=cb,
    ADbackend=AutoZygote(),
    hasconverged=checkconv,
)
losses = map(x -> x.loss, stats)
