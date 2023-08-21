using CUDA
using LinearAlgebra
using FunctionChains
using Bijectors
using Flux
using NormalizingFlows

rng = CUDA.default_rng()
T = Float32
q0 = MvNormal(ones(T, 2))
q0_g = MvNormal(CUDA.zeros(T, 2), I)

ts = reduce(∘, [f32(Bijectors.PlanarLayer(2)) for _ in 1:2])
flow = transformed(q0, ts)

# gpu 
CUDA.functional()
ts_g = gpu(ts)
flow_g = transformed(q0_g, ts_g)

xs = rand(rng, flow_g.dist, 10) # on cpu
ys_g = transform(ts_g, cu(xs)) # good
logpdf(flow_g, ys_g[:, 1]) # good
rand(flow_g) # bug

using CUDA
using LinearAlgebra
using Distributions, Random
using Bijectors

rng = CUDA.default_rng()
T = Float32
q0_g = MvNormal(CUDA.zeros(T, 2), I)

CUDA.functional()
ts_g = gpu(ts)
flow_g = transformed(q0_g, ts_g)

x = rand(rng, q0_g) # good 
xs = rand(rng, q0_g) # ambiguous 

y = rand_device(rng, flow_g, 10) # ambiguous 

rand(gpu(flow))

gpu(q0)
using Functors
@functor MvNormal
