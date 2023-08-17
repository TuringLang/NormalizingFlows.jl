# Defining Your Own Flow Layer

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
layer -- an `Affine Coupling Layer` (Dinh *et al.*, 2016) -- using `Bijectors.jl` and `Flux.jl`.

## Affine coupling flow

Given an input vector $\boldsymbol{x}$, the general *coupling transformation* splits it into two
parts: $\boldsymbol{x}_{I_1}$ and $\boldsymbol{x}_{I\setminus I_1}$. Only one
part (e.g., $\boldsymbol{x}_{I_1}$) undergoes a bijective transformation $f$, noted as the *coupling law*, 
based on the values of the other part (e.g., $\boldsymbol{x}_{I\setminus I_1}$), which remains unchanged. 
```math
\begin{array}{llll}
c_{I_1}(\cdot ; f, \theta): & \mathbb{R}^d \rightarrow \mathbb{R}^d & c_{I_1}^{-1}(\cdot ; f, \theta): & \mathbb{R}^d \rightarrow \mathbb{R}^d \\
& \boldsymbol{x}_{I \backslash I_1} \mapsto \boldsymbol{x}_{I \backslash I_1} & & \boldsymbol{y}_{I \backslash I_1} \mapsto \boldsymbol{y}_{I \backslash I_1} \\
& \boldsymbol{x}_{I_1} \mapsto f\left(\boldsymbol{x}_{I_1} ; \theta\left(\boldsymbol{x}_{I\setminus I_1}\right)\right) & & \boldsymbol{y}_{I_1} \mapsto f^{-1}\left(\boldsymbol{y}_{I_1} ; \theta\left(\boldsymbol{y}_{I\setminus I_1}\right)\right)
\end{array}
```
Here $\theta$ can be an arbitrary function, e.g., a neural network.
As long as $f(\cdot; \theta(\boldsymbol{x}_{I\setminus I_1}))$ is invertible, $c_{I_1}$ is invertible, and the 
Jacobian determinant of $c_{I_1}$ is easy to compute:
```math
\left|\text{det} \nabla_x c_{I_1}(x)\right| = \left|\text{det} \nabla_{x_{I_1}} f(x_{I_1}; \theta(x_{I\setminus I_1}))\right|
```

The affine coupling layer is a special case of the coupling transformation, where the coupling law $f$ is an affine function:
```math
\begin{aligned}
\boldsymbol{x}_{I_1} &\mapsto \boldsymbol{x}_{I_1} \odot s\left(\boldsymbol{x}_{I\setminus I_1}\right) + t\left(\boldsymbol{x}_{I \setminus I_1}\right) \\
\boldsymbol{x}_{I \backslash I_1} &\mapsto \boldsymbol{x}_{I \backslash I_1}
\end{aligned}
```
Here, $s$ and $t$ are arbitrary functions (often neural networks) called the
"scaling" and "translation" functions, respectively. They produce vectors of the
same dimension as $\boldsymbol{x}_{I_1}$.


## Implementing Affine Coupling Layer

We start by defining a simple 3-layer multi-layer perceptron (MLP) using `Flux.jl`, which will be 
used to define the scaling $s$ and translation functions $t$ in the affine coupling layer.
```@example afc
using Flux

function MLP_3layer(input_dim::Int, hdims::Int, output_dim::Int; activation=Flux.leakyrelu)
    return Chain(
        Flux.Dense(input_dim, hdims, activation),
        Flux.Dense(hdims, hdims, activation),
        Flux.Dense(hdims, output_dim),
    )
end
```

#### Construct the object

Following the user interface of `Bijectors.jl`, we define a struct `AffineCoupling` as a subtype of `Bijectors.Bijector`.
The functions `parition` , `combine` are used to partition and recombine a vector into 3 disjoint subvectors. 
And `PartitionMask` is used to store this partition rule. 
These three functions are
all defined in `Bijectors.jl`; see the [documentaion](https://github.com/TuringLang/Bijectors.jl/blob/49c138fddd3561c893592a75b211ff6ad949e859/src/bijectors/coupling.jl#L3) for more details.

```@example afc
using Functors
using Bijectors
using Bijectors: partition, combine, PartitionMask

struct AffineCoupling <: Bijectors.Bijector
    dim::Int
    mask::Bijectors.PartitionMask
    s::Flux.Chain
    t::Flux.Chain
end

# to apply functions to the parameters that are contained in AffineCoupling.s and AffineCoupling.t, 
# and to re-build the struct from the parameters, we use the functor interface of `Functors.jl` 
# see https://fluxml.ai/Flux.jl/stable/models/functors/#Functors.functor
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
By default, we define $s$ and $t$ using the `MLP_3layer` function, which is a
3-layer MLP with leaky ReLU activation function.

#### Implement the Forward and Inverse Transformations


```@example afc
function Bijectors.transform(af::AffineCoupling, x::AbstractVector)
    # partition vector using 'af.mask::PartitionMask`
    x₁, x₂, x₃ = partition(af.mask, x)
    y₁ = x₁ .* af.s(x₂) .+ af.t(x₂)
    return combine(af.mask, y₁, x₂, x₃)
end

function Bijectors.transform(iaf::Inverse{<:AffineCoupling}, y::AbstractVector)
    af = iaf.orig
    # partition vector using `af.mask::PartitionMask`
    y_1, y_2, y_3 = partition(af.mask, y)
    # inverse transformation
    x_1 = (y_1 .- af.t(y_2)) ./ af.s(y_2)
    return combine(af.mask, x_1, y_2, y_3)
end
```

#### Implement the Log-determinant of the Jacobian
Notice that here we wrap the transformation and the log-determinant of the Jacobian into a single function, `with_logabsdet_jacobian`.

```@example afc
function Bijectors.with_logabsdet_jacobian(af::AffineCoupling, x::AbstractVector)
    x_1, x_2, x_3 = Bijectors.partition(af.mask, x)
    y_1 = af.s(x_2) .* x_1 .+ af.t(x_2)
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
#### Construct Normalizing Flow

Now with all the above implementations, we are ready to use the `AffineCoupling` layer for normalizing flow 
by applying it to a base distribution $q_0$.

```@example afc
using Random, Distributions, LinearAlgebra
dim = 4
hdims = 10
Ls = [
    AffineCoupling(dim, hdims, 1:2), 
    AffineCoupling(dim, hdims, 3:4), 
    AffineCoupling(dim, hdims, 1:2), 
    AffineCoupling(dim, hdims, 3:4), 
    ]
ts = reduce(∘, Ls)
q₀ = MvNormal(zeros(Float32, dim), I)
flow = Bijectors.transformed(q₀, ts)
```
We can now sample from the flow:
```@example afc
x = rand(flow, 10)
```
And evaluate the density of the flow:
```@example afc
logpdf(flow, x[:,1])
```


## Reference
Dinh, L., Sohl-Dickstein, J. and Bengio, S., 2016. *Density estimation using real nvp.* 
arXiv:1605.08803.