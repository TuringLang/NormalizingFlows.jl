# 0.3.0

## Breaking changes

`nsf` now builds its rational quadratic splines with the batched spline in Bijectors, and MonotonicSplines is no longer a dependency. The flow trains under every supported AD backend and runs on the GPU.

The spline boundary `B` is now honoured when scaling the knots. The previous implementation always scaled the knots into (-5, 5] regardless of `B`, so flows built with any other `B` define a different transform than before.

## Other changes

`NSF_layer` rejects `dim < 2` and `NeuralSplineCoupling` validates its mask, instead of failing later with a `DivideError`.
