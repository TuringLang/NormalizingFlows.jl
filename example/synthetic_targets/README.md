# NormalizingFlows Model-Zoo 

This folder contains various demonstrations of the usage of the `NormalizingFlows.jl` package.

`targets/` contains a collection of example target distributions for testing/benchmarking normalizing flows.

Each `*_flow.jl` file provides a demonstration of how to train the corresponding
normalizing flow to approximate the target distribution using `NormalizingFlows.jl` package.

## Usage 

Currently, all examples share the same [Julia project](https://pkgdocs.julialang.org/v1/environments/#Using-someone-else's-project). To run the examples, first activate the project environment:

```julia
# pwd() = "NormalizingFlows.jl/"
using Pkg; Pkg.activate("example"); Pkg.instantiate()
```
This will install all needed packages, at the exact versions when the model was last updated. Then you can run the model code with include("<example-to-run>.jl"), or by running the example script line-by-line.

