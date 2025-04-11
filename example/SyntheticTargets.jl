using DocStringExtensions
using Distributions, Random, LinearAlgebra
using IrrationalConstants
using Plots


include("targets/banana.jl")
include("targets/cross.jl")
include("targets/neal_funnel.jl")
include("targets/warped_gaussian.jl")


function load_model(name::String)
    if name == "Banana"
        return Banana(2, 1.0, 10.0)
    elseif name == "Cross"
        return Cross()
    elseif name == "Funnel"
        return Funnel(2)
    elseif name == "WarpedGaussian"
        return WarpedGauss()
    else
        error("Model not defined")
    end
end

function visualize(p::ContinuousMultivariateDistribution, samples=rand(p, 1000))
    xrange = range(minimum(samples[1, :]) - 1, maximum(samples[1, :]) + 1; length=100)
    yrange = range(minimum(samples[2, :]) - 1, maximum(samples[2, :]) + 1; length=100)
    z = [exp(Distributions.logpdf(p, [x, y])) for x in xrange, y in yrange]
    fig = contour(xrange, yrange, z'; levels=15, color=:viridis, label="PDF", linewidth=2)
    scatter!(samples[1, :], samples[2, :]; label="Samples", alpha=0.3, legend=:bottomright)
    return fig
end
