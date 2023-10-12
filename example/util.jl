using LinearAlgebra
using StatsBase
using BlockBandedMatrices
using Flux, Bijectors
using Base.Threads
using DiffResults

function MLP_3layer(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim),
    )
end

# require batching input
function MLP_BN(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims),
        Flux.BatchNorm(hdims, activation; track_stats=true, affine=true),
        Flux.Dense(hdims, hdims, activation),
        Flux.BatchNorm(hdims, activation; track_stats=true, affine=true),
        Flux.Dense(hdims, output_dim),
        Flux.BatchNorm(output_dim, activation; track_stats=true, affine=true),
    )
end

# function resblock(inputdim::Int, hdim::Int, outputdim::Int; activation=Flux.leakyrelu)
#     mlp_layer = MLP_3layer(inputdim, hdim, outputdim; activation=activation)
#     return Flux.SkipConnection(mlp_layer, +)
#     # return Chain(res, BatchNorm(outputdim, activation; track_stats=true, affine=true))
# end
function resblock(input_dim::Int, hidden_dim::Int, output_dim::Int; activation=relu)
    mlp = Chain(
        Dense(input_dim, hidden_dim, activation),
        BatchNorm(hidden_dim),
        Dense(hidden_dim, output_dim),
    )
    return SkipConnection(mlp, +)
end

function resnet(
    input_dim::Int, hidden_dim::Int, output_dim::Int; activation=relu, n_blocks=2
)
    layers = [
        resblock(input_dim, hidden_dim, output_dim; activation=activation) for
        _ in 1:n_blocks
    ]
    return Chain(layers...)
end
# function resblockBN(inputdim::Int, hdim::Int, outputdim::Int; activation=Flux.leakyrelu)
#     mlp_layer = MLP_BN(inputdim, hdim, outputdim; activation=activation)
#     return Flux.SkipConnection(mlp_layer, +)
# end

function rand_batch(rng::AbstractRNG, td::Bijectors.MvTransformed, num_samples::Int)
    samples = rand(rng, td.dist, num_samples)
    res = td.transform(samples)
    return res
end
function rand_batch(td::Bijectors.MvTransformed, num_samples::Int)
    return rand_batch(Random.default_rng(), td, num_samples)
end

###############33
# functions for error 
###################

# this is how to extract functions 
function get_functions(ts)
    fs = FunctionChains._flatten_composed(ts)[1]
    return fs.fs
end

function intermediate_flows(ts, q0)
    flows = []
    fs = get_functions(ts)
    for i in 1:length(fs)
        push!(flows, Bijectors.transformed(q0, fchain(fs[1:i])))
    end
    return flows
end
function intermediate_lpdfs(ts, q0, fwd_samples)
    flows = intermediate_flows(ts, q0)
    @assert length(flows) == length(fwd_samples) "numder of layers and numbers of sample batches mismatch"
    lpdfs = Vector{Vector{eltype(fwd_samples[1])}}(undef, length(flows))
    @threads for i in 1:length(flows)
        flow = flows[i]
        ys = fwd_samples[i]
        lpdf = logpdf(flow, ys)
        lpdfs[i] = lpdf
    end
    return reduce(hcat, lpdfs)
end
function inverse_from_intermediate_layers(ts, fwd_samples)
    inv_ts = []
    fs = get_functions(ts)
    for i in 1:length(fs)
        it = inverse(fchain(fs[1:i]))
        push!(inv_ts, it)
    end

    @assert length(inv_ts) == length(fwd_samples) "numder of layers and numbers of sample batches mismatch"
    X0 = Vector{Matrix{eltype(fwd_samples[1])}}(undef, length(inv_ts))
    @threads for i in 1:length(inv_ts)
        f = inv_ts[i]
        ys = fwd_samples[i]
        x0 = f(ys)
        X0[i] = x0
    end
    return X0
end
function elbo_intermediate(ts, q0, logp, Xs)
    flows = intermediate_flows(ts, q0)
    Els = Vector{eltype(Xs)}(undef, length(flows))
    @threads for i in 1:length(flows)
        flow = flows[i]
        el = elbo_batch(flow, logp, Xs)
        Els[i] = el
    end
    return Els
end
function llh_intermediate(ts, q0, Xs)
    flows = intermediate_flows(ts, q0)
    Els = Vector{eltype(Xs)}(undef, length(flows))
    @threads for i in 1:length(flows)
        flow = flows[i]
        el = llh_batch(flow, Xs)
        Els[i] = el
    end
    return Els
end
function single_fwd_err(ts, fwd_sample_big, Xs)
    layers = get_functions(ts)
    fwd_sample_big32 = map(x -> Float32.(x), fwd_sample_big)
    diff = [layers[1](Xs) .- fwd_sample_big32[1]]
    for i in 2:length(layers)
        layer = layers[i]
        fwd_sample = layer(fwd_sample_big32[i - 1])
        push!(diff, fwd_sample .- fwd_sample_big32[i])
    end
    return diff
end
function single_bwd_err(its, bwd_sample_big, Ys)
    layers = get_functions(its)
    bwd_sample_big32 = map(x -> Float32.(x), bwd_sample_big)
    diff = [layers[1](Ys) .- bwd_sample_big32[1]]
    for i in 2:length(layers)
        layer = layers[i]
        bwd_sample = layer(bwd_sample_big32[i - 1])
        push!(diff, bwd_sample .- bwd_sample_big32[i])
    end
    return diff
end

function flow_jacobians(ts, x)
    layers = get_functions(ts)
    rs = DiffResults.JacobianResult(x)
    ft = eltype(x)
    Ms = []
    for i in 1:length(layers)
        l = Flux._paramtype(ft, layers[i])
        rs = ForwardDiff.jacobian!(rs, l, x)
        x, J = DiffResults.value(rs), DiffResults.jacobian(rs)
        push!(Ms, copy(J))
    end
    return Ms
end

# function flow_bwd_jacobians(its, one_bwd_sample)
#     layers = get_functions(its)
#     Ms = []
#     for i in 1:length(one_bwd_sample)
#         layer = layers[i]
#         J = ForwardDiff.jacobian(layer, one_bwd_sample[i])
#         push!(Ms, J)
#     end
#     return Ms
# end

function construct_shadow_matrix(M)
    Diag = [m * m' + I for m in M]
    offD = [-m for m in M[2:end]]
    L = BlockTridiagonal(offD, Diag, offD)
    return Symmetric(Matrix(L), :L)
end

function shadowing_window(L, δ)
    σ = sqrt(eigmin(L))
    return 2 * δ / σ
end

function all_shadowing_window(Ms::Tuple, δ)
    L0 = Symmetric(Ms[1] * Ms[1] + I, :L)
    w0 = [shadowing_window(L0, δ)]
    Ls = [construct_shadow_matrix(Ms[1:i]) for i in 2:length(Ms)]
    ws = [shadowing_window(L, δ) for L in Ls]
    return vcat(w0, ws)
end

function all_shadowing_window(ts::FunctionChain, x0, δ)
    Ms = flow_jacobians(ts, x0)
    return all_shadowing_window(Ms, δ)
end

# aux function for generating ribbon plot
function get_percentiles(dat; p1=25, p2=75, byrow=true)
    # if a single batch is listed by row, flip the datmat 
    if byrow
        dat = Matrix(dat')
    end
    n = size(dat, 2)

    plow = zeros(n)
    phigh = zeros(n)

    for i in 1:n
        dat_remove_nan = (dat[:, i])[iszero.(isnan.(dat[:, i]))]
        median_remove_nan = median(dat_remove_nan)
        plow[i] = median_remove_nan - percentile(vec(dat_remove_nan), p1)
        phigh[i] = percentile(vec(dat_remove_nan), p2) - median_remove_nan
    end

    return plow, phigh
end