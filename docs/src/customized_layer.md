## Customize your own flow layer

In practice, user might want to define their own normalizing flow. 
As briefly noted in [What are normalizing flows?](@ref), the key is to define a
customized normalizing flow layer, including its transformation and inverse,
as well as the log-determinant of the Jacobian of the transformation.
`Bijectors.jl` offers a convenient interface to define a customized bijection.
We refer users to [the documentation of
`Bijectors.jl`](https://turinglang.org/Bijectors.jl/dev/transforms/#Implementing-a-transformation)
for more details.
`Flux.jl` is also a useful package, offering a convenient interface to define neural networks.


In this tutorial, we demonstrate how to define a customized normalizing flow
layer --an **affine coupling layer** (Dinh *et al.*, 2016) -- using `Bijectors.jl` and `Flux.jl`.

### Example: Affine coupling layer

Given an input vector $\boldsymbol{x}$, the general *coupling transformation* splits it into two
parts: $\boldsymbol{x}_{I_1}$ and $\boldsymbol{x}_{I\setminus I_1}$. Only one part (e.g., $\boldsymbol{x}_{I_1}$) undergoes a 
bijective transformation $f$, noted as the *coupling law*,  
based on the values of the other part (e.g., $\boldsymbol{x}_{I\setminus I_1}$), which remains unchanged. 
```math
\begin{array}{llll}
c_{I_1}(\cdot ; f, \theta): & \mathbb{R}^d \rightarrow \mathbb{R}^d & c_{I_1}^{-1}(\cdot ; f, \theta): & \mathbb{R}^d \rightarrow \mathbb{R}^d \\
& \boldsymbol{x}_{I \backslash I_1} \mapsto \boldsymbol{x}_{I \backslash I_1} & & \boldsymbol{y}_{I \backslash I_1} \mapsto \boldsymbol{y}_{I \backslash I_1} \\
& \boldsymbol{x}_{I_1} \mapsto f\left(\boldsymbol{x}_{I_1} ; \theta\left(\boldsymbol{x}_{I\setminus I_1}\right)\right) & & \boldsymbol{y}_{I_1} \mapsto f^{-1}\left(\boldsymbol{y}_{I_1} ; \theta\left(\boldsymbol{y}_{I\setminus I_1}\right)\right)
\end{array}
```
As long as $f(\cdot; \theta(\boldsymbol{x}_{I\setminus I_1}))$ is invertible, $c_{I_1}$ is invertible.
Here $\theta$ can be a arbitrary complex function, e.g., a neural network.

The affine coupling layer is a special case of the coupling transformation, 
where the transformation $f$ is defined as an affine function:
```math
\begin{aligned}
\boldsymbol{x}_{I_1} &\mapsto \boldsymbol{x}_{I_1} \odot s\left(\boldsymbol{x}_{I\setminus I_1}\right) + t\left(\boldsymbol{x}_{I \setminus I_1}\right) \\
\boldsymbol{x}_{I \backslash I_1} &\mapsto \boldsymbol{x}_{I \backslash I_1}
\end{aligned}
```
Here, $s$ and $t$ are arbitrary functions (often neural networks) called the
"scaling" and "translation" functions, respectively. They produce vectors of the
same dimension as $\boldsymbol{x}_{I_1}$.



We start by defining a simple 3-layer multi-layer perceptron (MLP) using `Flux.jl`:
```@example afc
using Flux
using Functors
using Bijectors
using Bijectors: partition, combine, PartitionMask

function MLP_3layer(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim),
    )
end
```


```@example afc
struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# let params track field s and t
@functor AffineCoupling (s, t)

function AffineCoupling(
    dim::Int,  # dimension of input
    hdims::Int, # dimension of hidden units for s and t
    mask_idx::AbstractVector, # index of dimension that one wants to apply transformations on
)
    cdims = length(mask_idx) # dimension of parts used to construct coupling law
    s = MLP_3layer(cdims, hdims, cdims)
    t = MLP_3layer(cdims, hdims, cdims)
    mask = PartitionMask(dim, mask_idx)
    return AffineCoupling(dim, mask, s, t)
end
```


We need to define forward transformation and inverse of the layer.

```@example afc
function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃ = partition(af.mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return combine(af.mask, y₁, x₂, x₃)
end
```


Define `with_logabsdet_jacobian` 
```@example afc
function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = exp.(af.s(x_2)) .* x_1 .+ af.t(x_2)
    logjac = sum(log ∘ abs, af.s(x_2))
    return combine(af.mask, y_1, x_2, x_3), logjac
end

function Bijectors.with_logabsdet_jacobian(
    iaf::Inverse{<:AffineCoupling}, y::AbstractVector
)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    logjac = -sum(log ∘ abs, af.s(y_2))
    return combine(af.mask, x_1, y_2, y_3), logjac
end
```


## Reference
Dinh, L., Sohl-Dickstein, J. and Bengio, S., 2016. *Density estimation using real nvp.* 
arXiv:1605.08803.