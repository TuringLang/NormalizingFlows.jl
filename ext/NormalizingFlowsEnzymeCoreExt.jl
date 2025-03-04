module NormalizingFlowsEnzymeCoreExt

using EnzymeCore
using NormalizingFlows
using NormalizingFlows: ADTypes, DifferentiationInterface

# deal with Enzyme readonly error: see https://discourse.julialang.org/t/enzyme-autodiff-readonly-error-and-working-with-batches-of-data/123012
function NormalizingFlows._prepare_gradient(loss, adbackend::ADTypes.AutoEnzyme, θ, args...)
    if isempty(args)
        return DifferentiationInterface.prepare_gradient(
            EnzymeCore.Const(loss), adbackend, θ
        )
    end
    return DifferentiationInterface.prepare_gradient(
        EnzymeCore.Const(loss),
        adbackend,
        θ,
        map(DifferentiationInterface.Constant, args)...,
    )
end

function NormalizingFlows._value_and_gradient(
    loss, prep, adbackend::ADTypes.AutoEnzyme, θ, args...
)
    if isempty(args)
        return DifferentiationInterface.value_and_gradient(
            EnzymeCore.Const(loss), prep, adbackend, θ
        )
    end
    return DifferentiationInterface.value_and_gradient(
        EnzymeCore.Const(loss),
        prep,
        adbackend,
        θ,
        map(DifferentiationInterface.Constant, args)...,
    )
end

end
