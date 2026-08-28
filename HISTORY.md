# 0.4.1

## Other changes

`elbo_batch` now works when the samples live on a GPU.
It used to assemble the ELBO from `Distributions.logpdf`, which evaluates the base distribution column by column and returns a host array.
The base distribution's log-density is now taken with a whole-array form that stays on the device holding the samples, for any GPU array backend.
`elbo` is unchanged and still goes through `Distributions.logpdf`, so it stays CPU only.
Mooncake now differentiates the GPU path: it reads ChainRules rules one signature at a time, so the device draw carries a `Mooncake.@zero_derivative` declaration of its own in a new extension.

Coupling flows (`realnvp`, `nsf`) still do not run on the GPU.
`Bijectors.PartitionMask` holds host sparse matrices and `partition`/`combine` multiply against them, so the split is done on the host.
`example/gpu/demo_gpu.jl` trains a planar flow instead.

The base distribution is treated as a constant when its log-density is taken this way.
Differentiating more than one use of a full covariance leaves a cotangent per use, and summing those indexes a device array element by element.

# 0.4.0

## Breaking changes

DifferentiationInterface has been removed as a dependency. Automatic differentiation is now routed through AbstractPPL's evaluator interface (`AbstractPPL.prepare` and `AbstractPPL.value_and_gradient!!`), following the rest of the Turing ecosystem.

The AD backend package must now be loaded so that its `AbstractPPL.prepare` method is available:

  - `AutoForwardDiff` works with `using ForwardDiff`.
  - `AutoMooncake` works with `using Mooncake`.
  - Other backends routed through DifferentiationInterface (`AutoZygote`, `AutoReverseDiff`, `AutoEnzyme`) additionally require `using DifferentiationInterface` alongside the concrete backend package.

`AutoReverseDiff(; compile=true)` stays rejected, and the rejection now matters: through the evaluator interface a compiled tape does take effect and bakes the objective's context into itself, so the gradient would be taken against the first iteration's random draws. Under the previous DifferentiationInterface path the flag was silently dropped instead.

# 0.3.0

## Breaking changes

`nsf` now builds its rational quadratic splines with a batched implementation in this package, and MonotonicSplines is no longer a dependency. The spline is written with whole-array operations, so the flow runs on the GPU and trains under Zygote, ForwardDiff, ReverseDiff, Mooncake, and, on Julia 1.11 and newer, Enzyme.

The spline boundary `B` is now honoured when scaling the knots. The previous implementation always scaled the knots into (-5, 5] regardless of `B`, so flows built with any other `B` define a different transform than before.

Bin widths, heights, and derivatives now have a floor of 1e-3, so no bin or slope can underflow to zero and produce NaN. This shifts the constrained knots slightly, so the transform differs from previous releases at every `B`, `B = 5` included, and weights trained with an older version define a different flow.

`AutoReverseDiff(; compile=true)` is now rejected rather than silently ignored. DifferentiationInterface drops the compile flag when context arguments are present, which is how the objective's arguments are passed here, and a tape that did take effect would freeze the random number generator.

`NeuralSplineCoupling` requires `B > 0`. A negative boundary silently broke invertibility and zero silently made the layer the identity.

## Other changes

`NSF_layer` rejects `dim < 2` and `NeuralSplineCoupling` validates its mask, instead of failing later with a `DivideError`.
