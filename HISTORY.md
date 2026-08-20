# 0.3.0

## Breaking changes

`nsf` now builds its rational quadratic splines with a batched implementation in this package, and MonotonicSplines is no longer a dependency. The spline is written with whole-array operations, so the flow runs on the GPU and trains under Zygote, ForwardDiff, ReverseDiff, Mooncake, and, on Julia 1.11 and newer, Enzyme.

The spline boundary `B` is now honoured when scaling the knots. The previous implementation always scaled the knots into (-5, 5] regardless of `B`, so flows built with any other `B` define a different transform than before.

Bin widths, heights, and derivatives now have a floor of 1e-3, so no bin or slope can underflow to zero and produce NaN. This shifts the constrained knots slightly, so the transform differs from previous releases at every `B`, `B = 5` included, and weights trained with an older version define a different flow.

`AutoReverseDiff(; compile=true)` is now rejected. A compiled tape freezes the objective's constants, the random number generator among them, so every iteration differentiated against the draws of the first one.

`NeuralSplineCoupling` requires `B > 0`. A negative boundary silently broke invertibility and zero silently made the layer the identity.

## Other changes

`NSF_layer` rejects `dim < 2` and `NeuralSplineCoupling` validates its mask, instead of failing later with a `DivideError`.
