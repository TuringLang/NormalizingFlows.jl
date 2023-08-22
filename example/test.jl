using CUDA
using LinearAlgebra
using Distributions, Random
using Bijectors
using Flux
import NormalizingFlows as NF

CUDA.functional()
rng = CUDA.default_rng()
T = Float32
q0_g = MvNormal(CUDA.zeros(T, 2), I)
# construct gpu flow
ts = reduce(∘, [f32(Bijectors.PlanarLayer(2)) for _ in 1:2])
ts_g = gpu(ts)
flow_g = transformed(q0_g, ts_g)

# sample from GPU MvNormal
x = NF.rand_device(rng, q0_g) # good 
xs = NF.rand_device(rng, q0_g, 100) # ambiguous 

# sample from GPU flow
y = NF.rand_device(rng, flow_g) # ambiguous 
ys = NF.rand_device(rng, flow_g, 100) # ambiguous
