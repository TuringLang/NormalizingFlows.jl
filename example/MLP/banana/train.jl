using JLD2
include("setup.jl")
######################
# train the flow
#####################
function train(flow, data_load, opt, n_epoch)
    # destruct flow for explicit access to the parameters
    # use FunctionChains instead of simple compositions to construct the flow when many flow layers are involved
    # otherwise the compilation time for destructure will be too long
    θ_flat, re = Optimisers.destructure(flow)

    # Normalizing flow training loop 
    θ_flat_trained, opt_stats, st = NormalizingFlows.optimize(
        data_load,
        AutoZygote(),
        NormalizingFlows.llh_batch,
        θ_flat,
        re;
        n_epoch=n_epoch,
        optimiser=opt,
    )

    flow_trained = re(θ_flat_trained)
    return flow_trained, opt_stats, st
end

nepoch = 1500
opt = Optimisers.ADAM(9e-4)
flow_trained, stats, _ = train(flow, data_load, opt, nepoch)

θ_trained, re = Optimisers.destructure(flow_trained)

JLD2.save(
    "result/MLP.jld2",
    "model",
    flow_trained,
    "param",
    θ_trained,
    "opt_stat",
    stats,
    "nlayers",
    nlayers,
    "nepoch",
    nepoch,
    "data",
    data_load,
    "opt",
    opt,
    "target",
    p,
)
