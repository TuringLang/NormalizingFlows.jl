# 0.3.0

## Breaking changes

DifferentiationInterface has been removed as a dependency. Automatic differentiation is now routed through AbstractPPL's evaluator interface (`AbstractPPL.prepare` and `AbstractPPL.value_and_gradient!!`), following the rest of the Turing ecosystem.

For users, the consequence is that the AD backend package must be loaded so that its `AbstractPPL.prepare` method is available:

  - `AutoForwardDiff` works with `using ForwardDiff`.
  - `AutoMooncake` works with `using Mooncake`.
  - Other backends routed through DifferentiationInterface (`AutoZygote`, `AutoReverseDiff`, `AutoEnzyme`) additionally require `using DifferentiationInterface` alongside the concrete backend package.

## Other changes

Bijectors compat now includes 0.16 and CUDA compat now includes 6.
