using Bijectors: transformed
using Flux

"""
A simple wrapper for a 3 layer dense MLP
"""
function mlp3(input_dim::Int, hidden_dims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hidden_dims, activation),
        Flux.Dense(hidden_dims, hidden_dims, activation),
        Flux.Dense(hidden_dims, output_dim),
    )
end

function create_flow(Ls, q₀)
    ts =  reduce(∘, Ls)
    return transformed(q₀, ts)
end
