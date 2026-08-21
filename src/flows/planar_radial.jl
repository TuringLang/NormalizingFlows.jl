"""
    planarflow(q0, nlayers; paramtype = Float64)

Construct a Planar Flow by stacking `nlayers` `Bijectors.PlanarLayer` blocks
on top of a base distribution `q0`.

Arguments
- `q0::Distribution{Multivariate,Continuous}`: base distribution (e.g., `MvNormal(zeros(d), I)`).
- `nlayers::Int`: number of planar layers to compose.

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type (use `Float32` for GPU friendliness).

Returns
- `Bijectors.TransformedDistribution` representing the planar flow.

Example
- `q0 = MvNormal(zeros(2), I); flow = planarflow(q0, 10)`
- `x = rand(flow, 128); lp = logpdf(flow, x)`
"""
function planarflow(
    q0::Distribution{Multivariate,Continuous},  
    nlayers::Int;                   
    paramtype::Type{T} = Float64,   
) where {T<:AbstractFloat}
    dim = length(q0)
    Ls = [Flux._paramtype(paramtype, Bijectors.PlanarLayer(dim)) for _ in 1:nlayers]
    return create_flow(Ls, q0)
end


"""
    radialflow(q0, nlayers; paramtype = Float64)

Construct a Radial Flow by stacking `nlayers` `Bijectors.RadialLayer` blocks
on top of a base distribution `q0`.

Arguments
- `q0::Distribution{Multivariate,Continuous}`: base distribution (e.g., `MvNormal(zeros(d), I)`).
- `nlayers::Int`: number of radial layers to compose.

Keyword Arguments
- `paramtype::Type{T} = Float64`: parameter element type (use `Float32` for GPU friendliness).

Returns
- `Bijectors.TransformedDistribution` representing the radial flow.

Example
- `q0 = MvNormal(zeros(2), I); flow = radialflow(q0, 6)`
- `x = rand(flow); lp = logpdf(flow, x)`
"""
function radialflow(
    q0::Distribution{Multivariate,Continuous},  
    nlayers::Int;                   
    paramtype::Type{T} = Float64,   
) where {T<:AbstractFloat}
    dim = length(q0)
    Ls = [Flux._paramtype(paramtype, Bijectors.RadialLayer(dim)) for _ in 1:nlayers]
    return create_flow(Ls, q0)
end
