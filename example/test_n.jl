using Flux
using Bijectors
using Bijectors: partition, combine, PartitionMask

using Random, Distributions, LinearAlgebra
using Functors
using Optimisers, ADTypes
using Mooncake, Zygote, Enzyme, ADTypes
import NormalizingFlows as NF

import DifferentiationInterface as DI


pt = Float64
inputdim = 4
outputdim = 3

x = randn(pt, inputdim)

bs = 64
xs = randn(pt, inputdim, 64)

# compose two fully connected networks
m1 = NF.fnn(inputdim, [16, 16], outputdim; output_activation=nothing, paramtype=pt)
m2 = NF.fnn(outputdim, [16, 16], inputdim; output_activation=Flux.tanh, paramtype=pt)
mm = reduce(∘, (m2, m1))
psm, stm = Optimisers.destructure(mm)

function lsm(ps, st, x)
    model = st(ps)
    y = model(x)
    return sum(y) # just a dummy loss
end

adtype = ADTypes.AutoMooncake(; config = Mooncake.Config())

val, grad = DI.value_and_gradient(
    lsm, adtype, 
    psm, DI.Cache(stm), DI.Constant(xs)
)


acl = NF.AffineCoupling( inputdim, [16, 16], 1:2:inputdim, pt)
psacl,stacl = Optimisers.destructure(acl)

function loss(ps, st, x)
    model = st(ps)
    y = model(x)
    return sum(y) # just a dummy loss
end

val, grad = DI.value_and_gradient(
    loss, 
    ADTypes.AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    psacl, DI.Cache(stacl), DI.Constant(x)
)

# val, grad = DI.value_and_gradient(
#     loss, 
#     ADTypes.AutoMooncake(; config = Mooncake.Config()), 
#     psacl, DI.Cache(stacl), DI.Constant(x)
# )

function loss_acl_manual(ps, st, x)
    acl = st(ps)
    s_net = acl.s
    t_net = acl.t
    mask = acl.mask
    x₁, x₂, x₃ = partition(mask, x)
    y₁ = exp.(s_net(x₂)) .* x₁ .+ t_net(x₂)
    y = combine(mask, y₁, x₂, x₃)
    # println("y = ", y)
    return sum(y)
end

val, grad = DI.value_and_gradient(
    loss_acl_manual, 
    # ADTypes.AutoMooncake(; config = Mooncake.Config()), 
    # ADTypes.AutoEnzyme(;
    #         mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    #         function_annotation=Enzyme.Const,
    #     ),
    psacl, DI.Cache(stacl), DI.Constant(x)
)



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

function ls_msk(ps, st, x, mask)
    t_net = st(ps)
    x₁, x₂, x₃ = partition(mask, x)
    y₁ = x₁ .+ t_net(x₂)
    y = combine(mask, y₁, x₂, x₃)
    # println("y = ", y)
    return sum(abs2, y)
end

inputdim = 4
mask_idx = 1:2:inputdim
mask = PartitionMask(inputdim, mask_idx)
cdim = length(mask_idx)

x = randn(inputdim)

t_net = mlp3(cdim, 16, cdim; paramtype = Float64)
ps, st = Optimisers.destructure(t_net)

ls_msk(ps, st, x, mask) # 3.0167880799441793

val, grad = DI.value_and_gradient(
    ls_msk, 
    ADTypes.AutoMooncake(; config = Mooncake.Config()), 
    ps, DI.Cache(st), DI.Constant(x), DI.Constant(mask)
)


struct ACL
    mask::Bijectors.PartitionMask
    t::Flux.Chain
end
@functor ACL (t, )

acl = ACL(mask, t_net)
psacl, stacl = Optimisers.destructure(acl)

function loss_acl(ps, st, x)
    acl = st(ps)
    t_net = acl.t
    mask = acl.mask
    x₁, x₂, x₃ = partition(mask, x)
    y₁ = x₁ .+ t_net(x₂)
    y = combine(mask, y₁, x₂, x₃)
    return sum(abs2, y)
end
loss_acl(psacl, stacl, x) # 3.0167880799441793

val, grad = DI.value_and_gradient(
    loss_acl, 
    ADTypes.AutoEnzyme(;
            mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
            function_annotation=Enzyme.Const,
        ),
    psacl, DI.Cache(stacl), DI.Constant(x)
)

val, grad = DI.value_and_gradient(
    loss_acl, 
    ADTypes.AutoMooncake(; config = Mooncake.Config()), 
    psacl, DI.Cache(stacl), DI.Constant(x)
)
