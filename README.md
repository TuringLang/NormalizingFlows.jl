# NormalizingFlows.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://turinglang.github.io/NormalizingFlows.jl/dev/)
[![Build Status](https://github.com/TuringLang/NormalizingFlows.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/TuringLang/NormalizingFlows.jl/actions/workflows/CI.yml?query=branch%3Amain)


A normalizing flow library for Julia.

The purpose of this package is to provide a simple and flexible interface for 
variational inference (VI) and normalizing flows (NF) for Bayesian computation or generative modeling.
The key focus is to ensure modularity and extensibility, so that users can easily 
construct (e.g., define customized flow layers) and combine various components 
(e.g., choose different VI objectives or gradient estimates) 
for variational approximation of general target distributions, 
without being tied to specific probabilistic programming frameworks or applications. 

See the [documentation](https://turinglang.org/NormalizingFlows.jl/dev/) for more.  

## Installation
To install the package, run the following command in the Julia REPL:
```julia
# install the package
]  # enter Pkg mode
(@v1.9) pkg> add git@github.com:TuringLang/NormalizingFlows.jl.git
```
Then simply run the following command to use the package:
```julia
using NormalizingFlows
```


## Quick recap of normalizing flows


## Current status and TODOs


