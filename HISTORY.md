# 0.3.0

## Breaking changes

`nsf` now builds its rational quadratic splines with a batched implementation in this package, and MonotonicSplines is no longer a dependency. The spline is written with whole-array operations, so the flow runs on the GPU and trains under Zygote, ForwardDiff, ReverseDiff, Mooncake, and, on Julia 1.11 and newer, Enzyme.

The spline boundary `B` is now honoured when scaling the knots. The previous implementation always scaled the knots into (-5, 5] regardless of `B`, so flows built with any other `B` define a different transform than before.

## Other changes

`NSF_layer` rejects `dim < 2` and `NeuralSplineCoupling` validates its mask, instead of failing later with a `DivideError`.
