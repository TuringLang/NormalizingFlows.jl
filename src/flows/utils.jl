using Bijectors: transformed
using Flux

"""
    mlp3(input_dim::Int, hidden_dims::Int, output_dim::Int; activation=Flux.leakyrelu)

A simple wrapper for a 3 layer dense MLP
"""
function mlp3(
    input_dim::Int, 
    hidden_dims::Int, 
    output_dim::Int; 
    activation=Flux.leakyrelu,
    paramtype::Type{T} = Float64
) where {T<:AbstractFloat}
    m = Chain(
        Flux.Dense(input_dim, hidden_dims, activation),
        Flux.Dense(hidden_dims, hidden_dims, activation),
        Flux.Dense(hidden_dims, output_dim),
    )
    return Flux._paramtype(paramtype, m)
end

"""
    fnn(
        input_dim::Int,
        hidden_dims::AbstractVector{Int},
        output_dim::Int;
        inlayer_activation=Flux.leakyrelu,
        output_activation=nothing,
        paramtype::Type{T} = Float64,
    )

Create a fully connected neural network (FNN).

# Arguments
- `input_dim::Int`: The dimension of the input layer.
- `hidden_dims::AbstractVector{<:Int}`: A vector of integers specifying the dimensions of the hidden layers.
- `output_dim::Int`: The dimension of the output layer.
- `inlayer_activation`: The activation function for the hidden layers. Defaults to `Flux.leakyrelu`.
- `output_activation`: The activation function for the output layer. Defaults to `Flux.tanh`.
- `paramtype::Type{T} = Float64`: The type of the parameters in the network, defaults to `Float64`.

# Returns
- A `Flux.Chain` representing the FNN.
"""
function fnn(
    input_dim::Int,
    hidden_dims::AbstractVector{Int},
    output_dim::Int;
    inlayer_activation=Flux.leakyrelu,
    output_activation=nothing,
    paramtype::Type{T} = Float64,
) where {T<:AbstractFloat}
    # Create a chain of dense layers
    # First layer
    layers = Any[Flux.Dense(input_dim, hidden_dims[1], inlayer_activation)]

    # Hidden layers
    for i in 1:(length(hidden_dims) - 1)
        push!(
            layers,
            Flux.Dense(hidden_dims[i], hidden_dims[i + 1], inlayer_activation),
        )
    end

    # Output layer
    if output_activation === nothing
        push!(layers, Flux.Dense(hidden_dims[end], output_dim))
    else
        push!(layers, Flux.Dense(hidden_dims[end], output_dim, output_activation))
    end

    m = Chain(layers...)
    return Flux._paramtype(paramtype, m)
end

function create_flow(Ls, q₀)
    ts =  reduce(∘, Ls)
    return transformed(q₀, ts)
end