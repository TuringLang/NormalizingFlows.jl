using Distributions, Random, LinearAlgebra
using LogDensityProblems
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

LogDensityProblems.dimension(dist::ContinuousDistribution) = length(dist)
LogDensityProblems.logdensity(dist::ContinuousDistribution, x) = logpdf(dist, x)

