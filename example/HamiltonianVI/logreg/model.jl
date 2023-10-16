using Distributions, ForwardDiff, LinearAlgebra, Random
using Base.Threads: @threads
using Base.Threads
using Tullio, Zygote
using Zygote: @adjoint, refresh
using DataFrames, CSV

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
###########
## load dataset 
###########
# dat = load("data/dataset.jld")
# X, Y = dat["X"], dat["Y"]

df = DataFrame(CSV.File("data/final_dat.csv"))
xs = Matrix(df)[:, 2:end]
N = size(xs, 1)
X_raw = xs[:, 1:(end - 1)]
const X = (X_raw .- mean(X_raw; dims=1)) ./ std(X_raw; dims=1)
const Y = xs[:, end]

const p = size(X, 2)
const d = p + 1
const aa, bb = 1.0, 0.01

# const X_big = bf.(X)
# const Y_big = bf.(Y)
# const aa_big, bb_big = bf(aa), bf(bb)
##########
# log posterior
##########
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
function logpost(θ::AbstractVector, X, Y, aa, bb, p)
    τ = θ[1]
    W = @view(θ[2:end])
    Z = X * W
    logpτ = aa * τ - bb * exp(τ)
    logpW = p * τ / 2 - exp(τ) / 2 * sum(abs2, W)
    @tullio llh := (Y[n] - 1) * Z[n] + log_sigmoid(Z[n])
    return logpτ + logpW + llh
end
function logpost(θ::AbstractMatrix, X, Y, aa, bb, p)
    τ = θ[1, :]
    W = @view(θ[2:end, :])
    Z = X * W
    logpτ = @. aa * τ - bb * exp(τ)
    logpW = p .* τ ./ 2 - exp.(τ) ./ 2 .* vec(sum(abs2, W; dims=1))
    @tullio llh[i] := (Y[n] - 1) * Z[n, i] + log_sigmoid(Z[n, i])
    return logpτ + logpW + llh
end

function ∇logp(z::AbstractVector, X, Y, aa, bb, p)
    τ = z[1]
    W = @view(z[2:end])
    grad = similar(z)
    grad[1] = aa - bb * exp(τ) + p / 2 - exp(τ) / 2 * sum(abs2, W)
    S = neg_sigmoid.(X * W)
    @tullio M[j] := X[n, j] * (S[n] + Y[n])
    grad[2:end] .= -exp(τ) .* W .+ M
    return grad
end
function ∇logp(z::AbstractMatrix, X, Y, aa, bb, p)
    τ = z[1, :]
    W = @view(z[2:end, :])
    grad = similar(z)
    grad[1, :] = aa .- bb .* exp.(τ) .+ p / 2 .- exp.(τ) ./ 2 .* vec(sum(abs2, W; dims=1))
    S = neg_sigmoid.(X * W) .+ Y
    @tullio M[j, i] := X[n, j] * S[n, i]
    grad[2:end, :] .= -exp.(τ)' .* W .+ M
    return grad
end
# ∇logp(zs::AbstractMatrix) = reduce(hcat, map(∇logp, eachcol(zs))

logp(x) = logpost(x, X, Y, aa, bb, p)
∇S(x) = ∇logp(x, X, Y, aa, bb, p)
