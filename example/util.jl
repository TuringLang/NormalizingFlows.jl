using LinearAlgebra
using StatsBase
using BlockBandedMatrices
using Flux, Bijectors
using Base.Threads
using DiffResults
using ForwardDiff
using ProgressMeter
using TickTock

function MLP_3layer(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim),
    )
end

# require batching input
# function MLP_BN(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
#     return Chain(
#         Flux.Dense(input_dim, hdims),
#         Flux.BatchNorm(hdims, activation; track_stats=true, affine=true),
#         Flux.Dense(hdims, hdims, activation),
#         Flux.BatchNorm(hdims, activation; track_stats=true, affine=true),
#         Flux.Dense(hdims, output_dim),
#         Flux.BatchNorm(output_dim, activation; track_stats=true, affine=true),
#     )
# end

# # function resblock(inputdim::Int, hdim::Int, outputdim::Int; activation=Flux.leakyrelu)
# #     mlp_layer = MLP_3layer(inputdim, hdim, outputdim; activation=activation)
# #     return Flux.SkipConnection(mlp_layer, +)
# #     # return Chain(res, BatchNorm(outputdim, activation; track_stats=true, affine=true))
# # end
# function resblock(input_dim::Int, hidden_dim::Int, output_dim::Int; activation=relu)
#     mlp = Chain(
#         Dense(input_dim, hidden_dim, activation),
#         BatchNorm(hidden_dim),
#         Dense(hidden_dim, output_dim),
#     )
#     return SkipConnection(mlp, +)
# end

# function resnet(
#     input_dim::Int, hidden_dim::Int, output_dim::Int; activation=relu, n_blocks=2
# )
#     layers = [
#         resblock(input_dim, hidden_dim, output_dim; activation=activation) for
#         _ in 1:n_blocks
#     ]
#     return Chain(layers...)
# end
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
